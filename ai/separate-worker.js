"use strict";
/* ============================================================
   AI stem-separation worker for the karaoke app.

   Runs Meta's Demucs v4 (Hybrid Transformer, 4 stems) in the
   browser via onnxruntime-web (WebGPU when available, WASM
   fallback). The DSP + inference pipeline below is adapted from
   bakkot/demucs-js (MIT license, https://github.com/bakkot/demucs-js),
   which itself ports facebookresearch/demucs (MIT).
   Model weights: see ai/MODEL-LICENSE.md (personal/research use).

   Protocol (postMessage):
     in : { cmd:"separate", channelData:[Float32Array,Float32Array],
            sampleRate, base }   // base = URL prefix for ai/ assets
     out: { type:"download", loaded, total }        // model fetch
          { type:"progress", done, total }          // per chunk
          { type:"done", wav:Uint8Array, backend }  // instrumental WAV
          { type:"error", message }
   ============================================================ */

/* ---------------- FFT / STFT (from demucs-js dsp.ts) ---------------- */
function fft(realInput, imagInput = null) {
  const n = realInput.length;
  if ((n & (n - 1)) !== 0) throw new Error("FFT size must be power of 2");
  const real = new Float32Array(realInput);
  const imag = imagInput ? new Float32Array(imagInput) : new Float32Array(n);
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let k = n >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      let wr = 1, wi = 0;
      for (let jj = 0; jj < halfLen; jj++) {
        const k = i + jj;
        const l = k + halfLen;
        const tr = wr * real[l] - wi * imag[l];
        const ti = wr * imag[l] + wi * real[l];
        real[l] = real[k] - tr;
        imag[l] = imag[k] - ti;
        real[k] += tr;
        imag[k] += ti;
        const wtemp = wr;
        wr = wtemp * Math.cos(angle) - wi * Math.sin(angle);
        wi = wtemp * Math.sin(angle) + wi * Math.cos(angle);
      }
    }
  }
  return { real, imag };
}
function ifft(realInput, imagInput) {
  const n = realInput.length;
  const imagConj = new Float32Array(n);
  for (let i = 0; i < n; i++) imagConj[i] = -imagInput[i];
  const result = fft(realInput, imagConj);
  for (let i = 0; i < n; i++) {
    result.real[i] /= n;
    result.imag[i] = -result.imag[i] / n;
  }
  return result;
}
function hannWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / n));
  return w;
}
function padReflect(arr, padLeft, padRight) {
  const length = arr.length;
  const result = new Float32Array(length + padLeft + padRight);
  for (let i = 0; i < length; i++) result[padLeft + i] = arr[i];
  for (let i = 0; i < padLeft; i++) result[i] = result[2 * padLeft - i];
  for (let i = 0; i < padRight; i++)
    result[padLeft + length + i] = result[padLeft + length - 2 - i];
  return result;
}
function pad1d(x, paddings, mode = "constant") {
  const length = x.shape[x.shape.length - 1];
  const [paddingLeft, paddingRight] = paddings;
  let xData = x.data, xShape = x.shape;
  let actualPaddingLeft = paddingLeft, actualPaddingRight = paddingRight;
  if (mode === "reflect") {
    const maxPad = Math.max(paddingLeft, paddingRight);
    if (length <= maxPad) {
      const extraPad = maxPad - length + 1;
      const extraPadRight = Math.min(paddingRight, extraPad);
      const extraPadLeft = extraPad - extraPadRight;
      const shape = [...xShape];
      shape[shape.length - 1] = length + extraPadLeft + extraPadRight;
      const totalSize = shape.reduce((a, b) => a * b, 1);
      const tempData = new Float32Array(totalSize);
      const newLength = shape[shape.length - 1];
      const outerSize = totalSize / newLength;
      for (let i = 0; i < outerSize; i++)
        for (let j = 0; j < length; j++)
          tempData[i * newLength + extraPadLeft + j] = xData[i * length + j];
      xData = tempData; xShape = shape;
      actualPaddingLeft = paddingLeft - extraPadLeft;
      actualPaddingRight = paddingRight - extraPadRight;
    }
  }
  const currentLength = xShape[xShape.length - 1];
  const shape = [...xShape];
  shape[shape.length - 1] = currentLength + actualPaddingLeft + actualPaddingRight;
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(totalSize);
  const outputLength = shape[shape.length - 1];
  const outerSize = totalSize / outputLength;
  for (let i = 0; i < outerSize; i++) {
    const srcOffset = i * currentLength;
    const dstOffset = i * outputLength;
    if (mode === "constant") {
      for (let j = 0; j < currentLength; j++)
        out[dstOffset + actualPaddingLeft + j] = xData[srcOffset + j];
    } else {
      const slice = xData.slice(srcOffset, srcOffset + currentLength);
      const padded = padReflect(slice, actualPaddingLeft, actualPaddingRight);
      for (let j = 0; j < outputLength; j++) out[dstOffset + j] = padded[j];
    }
  }
  return { data: out, shape };
}
function stft(x, nFft, hopLength, window, normalized = true, center = true, padMode = "reflect") {
  const [batch, length] = x.shape;
  let inputData = x.data, inputLength = length;
  if (center) {
    const pad = Math.floor(nFft / 2);
    const padded = pad1d(x, [pad, pad], padMode);
    inputData = padded.data;
    inputLength = padded.shape[padded.shape.length - 1];
  }
  const numFrames = Math.floor((inputLength - nFft) / hopLength) + 1;
  const numFreqs = Math.floor(nFft / 2) + 1;
  const realOut = new Float32Array(batch * numFreqs * numFrames);
  const imagOut = new Float32Array(batch * numFreqs * numFrames);
  const norm = normalized ? 1.0 / Math.sqrt(nFft) : 1.0;
  for (let b = 0; b < batch; b++) {
    for (let frame = 0; frame < numFrames; frame++) {
      const frameStart = frame * hopLength;
      const frameData = new Float32Array(nFft);
      for (let i = 0; i < nFft; i++)
        frameData[i] = inputData[b * inputLength + frameStart + i] * window[i] * norm;
      const { real, imag } = fft(frameData);
      for (let freq = 0; freq < numFreqs; freq++) {
        const outIdx = b * numFreqs * numFrames + freq * numFrames + frame;
        realOut[outIdx] = real[freq];
        imagOut[outIdx] = imag[freq];
      }
    }
  }
  return {
    real: { data: realOut, shape: [batch, numFreqs, numFrames] },
    imag: { data: imagOut, shape: [batch, numFreqs, numFrames] },
  };
}
function spectro(x, nFft = 512, hopLength = null) {
  if (hopLength === null) hopLength = Math.floor(nFft / 4);
  const originalShape = x.shape;
  const length = originalShape[originalShape.length - 1];
  const otherSize = x.data.length / length;
  const window = hannWindow(nFft);
  const result = stft({ data: x.data, shape: [otherSize, length] }, nFft, hopLength, window, true, true, "reflect");
  const [batch, freqs, frames] = result.real.shape;
  const outputShape = [...originalShape.slice(0, -1), freqs, frames];
  return {
    real: { data: result.real.data, shape: outputShape },
    imag: { data: result.imag.data, shape: outputShape },
  };
}
function spec(x) {
  const hl = 1024;
  const lastDimLength = x.shape[x.shape.length - 1];
  const le = Math.ceil(lastDimLength / hl);
  const pad = Math.floor(hl / 2) * 3;
  const paddingRight = pad + le * hl - lastDimLength;
  const paddedX = pad1d(x, [pad, paddingRight], "reflect");
  const z = spectro(paddedX, 4096, hl);
  const zShape = z.real.shape;
  const newFreqs = zShape[zShape.length - 2] - 1;
  const frames = zShape[zShape.length - 1];
  const outerSize = z.real.data.length / (zShape[zShape.length - 2] * frames);
  const newShape = [...zShape];
  newShape[newShape.length - 2] = newFreqs;
  const newSize = newShape.reduce((a, b) => a * b, 1);
  const newReal = new Float32Array(newSize);
  const newImag = new Float32Array(newSize);
  for (let i = 0; i < outerSize; i++)
    for (let freq = 0; freq < newFreqs; freq++)
      for (let frame = 0; frame < frames; frame++) {
        const oldIdx = i * zShape[zShape.length - 2] * frames + freq * frames + frame;
        const newIdx = i * newFreqs * frames + freq * frames + frame;
        newReal[newIdx] = z.real.data[oldIdx];
        newImag[newIdx] = z.imag.data[oldIdx];
      }
  const slicedShape = [...newShape];
  slicedShape[slicedShape.length - 1] = le;
  const slicedSize = slicedShape.reduce((a, b) => a * b, 1);
  const slicedReal = new Float32Array(slicedSize);
  const slicedImag = new Float32Array(slicedSize);
  for (let i = 0; i < outerSize; i++)
    for (let freq = 0; freq < newFreqs; freq++)
      for (let frame = 0; frame < le; frame++) {
        const srcIdx = i * newFreqs * frames + freq * frames + (2 + frame);
        const dstIdx = i * newFreqs * le + freq * le + frame;
        slicedReal[dstIdx] = newReal[srcIdx];
        slicedImag[dstIdx] = newImag[srcIdx];
      }
  return {
    real: { data: slicedReal, shape: slicedShape },
    imag: { data: slicedImag, shape: slicedShape },
  };
}
function magnitude(z) {
  const [B, C, Fr, T] = z.real.shape;
  const out = new Float32Array(B * C * 2 * Fr * T);
  for (let b = 0; b < B; b++)
    for (let c = 0; c < C; c++)
      for (let fr = 0; fr < Fr; fr++)
        for (let t = 0; t < T; t++) {
          const srcIdx = b * C * Fr * T + c * Fr * T + fr * T + t;
          out[b * C * 2 * Fr * T + c * 2 * Fr * T + fr * T + t] = z.real.data[srcIdx];
          out[b * C * 2 * Fr * T + (c * 2 + 1) * Fr * T + fr * T + t] = z.imag.data[srcIdx];
        }
  return { data: out, shape: [B, C * 2, Fr, T] };
}
function istft(z, nFft, hopLength, window, normalized = true, length = null, center = true) {
  const [batch, freqs, numFrames] = z.real.shape;
  if (2 * freqs - 2 !== nFft) throw new Error("Expected freqs = nFft/2 + 1");
  let outputLength = length !== null ? length : nFft + (numFrames - 1) * hopLength;
  if (center) outputLength -= nFft;
  const output = new Float32Array(batch * outputLength);
  const windowSum = new Float32Array(batch * outputLength);
  const norm = normalized ? Math.sqrt(nFft) : 1.0;
  for (let b = 0; b < batch; b++) {
    for (let frame = 0; frame < numFrames; frame++) {
      const fullReal = new Float32Array(nFft);
      const fullImag = new Float32Array(nFft);
      for (let freq = 0; freq < freqs; freq++) {
        const idx = b * freqs * numFrames + freq * numFrames + frame;
        fullReal[freq] = z.real.data[idx];
        fullImag[freq] = z.imag.data[idx];
      }
      for (let freq = freqs; freq < nFft; freq++) {
        const mirrorFreq = nFft - freq;
        fullReal[freq] = fullReal[mirrorFreq];
        fullImag[freq] = -fullImag[mirrorFreq];
      }
      const frameData = ifft(fullReal, fullImag);
      const frameStart = frame * hopLength - (center ? nFft / 2 : 0);
      for (let i = 0; i < nFft; i++) {
        const outputIdx = frameStart + i;
        if (outputIdx >= 0 && outputIdx < outputLength) {
          const globalIdx = b * outputLength + outputIdx;
          output[globalIdx] += frameData.real[i] * window[i] * norm;
          windowSum[globalIdx] += window[i] * window[i];
        }
      }
    }
  }
  for (let i = 0; i < output.length; i++)
    if (windowSum[i] > 1e-8) output[i] /= windowSum[i];
  return { data: output, shape: [batch, outputLength] };
}
function ispectro(z, hopLength = null, length = null) {
  const originalShape = z.real.shape;
  const freqs = originalShape[originalShape.length - 2];
  const frames = originalShape[originalShape.length - 1];
  const nFft = 2 * freqs - 2;
  if (hopLength === null) hopLength = Math.floor(nFft / 4);
  const otherSize = z.real.data.length / (freqs * frames);
  const window = hannWindow(nFft);
  const result = istft(
    { real: { data: z.real.data, shape: [otherSize, freqs, frames] },
      imag: { data: z.imag.data, shape: [otherSize, freqs, frames] } },
    nFft, hopLength, window, true, length, true);
  const outputShape = [...originalShape.slice(0, -2), result.shape[1]];
  return { data: result.data, shape: outputShape };
}
function padComplex(z, padFreq, padTime) {
  const shape = z.real.shape;
  const ndim = shape.length;
  const [padFreqLeft, padFreqRight] = padFreq;
  const [padTimeLeft, padTimeRight] = padTime;
  const oldFreqs = shape[ndim - 2], oldFrames = shape[ndim - 1];
  const newFreqs = oldFreqs + padFreqLeft + padFreqRight;
  const newFrames = oldFrames + padTimeLeft + padTimeRight;
  const newShape = [...shape];
  newShape[ndim - 2] = newFreqs;
  newShape[ndim - 1] = newFrames;
  const totalSize = newShape.reduce((a, b) => a * b, 1);
  const newReal = new Float32Array(totalSize);
  const newImag = new Float32Array(totalSize);
  const outerSize = totalSize / (newFreqs * newFrames);
  for (let i = 0; i < outerSize; i++)
    for (let freq = 0; freq < oldFreqs; freq++)
      for (let frame = 0; frame < oldFrames; frame++) {
        const oldIdx = i * oldFreqs * oldFrames + freq * oldFrames + frame;
        const newIdx = i * newFreqs * newFrames + (freq + padFreqLeft) * newFrames + (frame + padTimeLeft);
        newReal[newIdx] = z.real.data[oldIdx];
        newImag[newIdx] = z.imag.data[oldIdx];
      }
  return {
    real: { data: newReal, shape: newShape },
    imag: { data: newImag, shape: newShape },
  };
}
function ispec(z, length) {
  const hl = 1024;
  let paddedZ = padComplex(z, [0, 1], [0, 0]);
  paddedZ = padComplex(paddedZ, [0, 0], [2, 2]);
  const pad = Math.floor(hl / 2) * 3;
  const le = hl * Math.ceil(length / hl) + 2 * pad;
  const x = ispectro(paddedZ, hl, le);
  const shape = x.shape;
  const lastDim = shape.length - 1;
  const totalLength = shape[lastDim];
  const newShape = [...shape];
  newShape[lastDim] = length;
  const totalSize = newShape.reduce((a, b) => a * b, 1);
  const newData = new Float32Array(totalSize);
  const outerSize = totalSize / length;
  for (let i = 0; i < outerSize; i++)
    for (let j = 0; j < length; j++)
      newData[i * length + j] = x.data[i * totalLength + pad + j];
  return { data: newData, shape: newShape };
}

/* ---------------- apply pipeline (from demucs-js apply.ts) ---------------- */
class TensorChunk {
  constructor(tensor, offset = 0, length = null) {
    const totalLength = tensor.shape[tensor.shape.length - 1];
    if (length === null) length = totalLength - offset;
    else length = Math.min(totalLength - offset, length);
    this.tensor = tensor;
    this.offset = offset;
    this.length = length;
  }
  padded(targetLength) {
    const delta = targetLength - this.length;
    const totalLength = this.tensor.shape[this.tensor.shape.length - 1];
    const start = this.offset - Math.floor(delta / 2);
    const end = start + targetLength;
    const correctStart = Math.max(0, start);
    const correctEnd = Math.min(totalLength, end);
    const padLeft = correctStart - start;
    const shape = [...this.tensor.shape];
    shape[shape.length - 1] = targetLength;
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(totalSize);
    const sliceLength = correctEnd - correctStart;
    const innerSize = targetLength;
    const outerSize = totalSize / innerSize;
    for (let i = 0; i < outerSize; i++) {
      const srcOffset = i * totalLength + correctStart;
      const dstOffset = i * innerSize + padLeft;
      for (let j = 0; j < sliceLength; j++)
        out[dstOffset + j] = this.tensor.data[srcOffset + j];
    }
    return { data: out, shape };
  }
}
function padToTrainingLength(mix, trainingLength) {
  const currentLength = mix.shape[mix.shape.length - 1];
  if (currentLength >= trainingLength) return mix;
  const shape = [...mix.shape];
  shape[shape.length - 1] = trainingLength;
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(totalSize);
  const outerSize = totalSize / trainingLength;
  for (let i = 0; i < outerSize; i++)
    for (let j = 0; j < currentLength; j++)
      out[i * trainingLength + j] = mix.data[i * currentLength + j];
  return { data: out, shape };
}
function maskToComplex(m) {
  const [B, S, C, Fr, T] = m.shape;
  const C_half = C / 2;
  const outShape = [B, S, C_half, Fr, T];
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const realOut = new Float32Array(outSize);
  const imagOut = new Float32Array(outSize);
  for (let b = 0; b < B; b++)
    for (let s = 0; s < S; s++)
      for (let c = 0; c < C_half; c++)
        for (let fr = 0; fr < Fr; fr++)
          for (let t = 0; t < T; t++) {
            const realInIdx = b * S * C * Fr * T + s * C * Fr * T + (c * 2) * Fr * T + fr * T + t;
            const imagInIdx = b * S * C * Fr * T + s * C * Fr * T + (c * 2 + 1) * Fr * T + fr * T + t;
            const outIdx = b * S * C_half * Fr * T + s * C_half * Fr * T + c * Fr * T + fr * T + t;
            realOut[outIdx] = m.data[realInIdx];
            imagOut[outIdx] = m.data[imagInIdx];
          }
  return {
    real: { data: realOut, shape: outShape },
    imag: { data: imagOut, shape: outShape },
  };
}
function addTensors(a, b) {
  const out = new Float32Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) out[i] = a.data[i] + b.data[i];
  return { data: out, shape: [...a.shape] };
}
function cropToValidLength(x, validLength) {
  const shape = x.shape;
  const currentLength = shape[shape.length - 1];
  if (validLength >= currentLength) return x;
  const newShape = [...shape];
  newShape[newShape.length - 1] = validLength;
  const totalSize = newShape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(totalSize);
  const outerSize = totalSize / validLength;
  for (let i = 0; i < outerSize; i++)
    for (let j = 0; j < validLength; j++)
      out[i * validLength + j] = x.data[i * currentLength + j];
  return { data: out, shape: newShape };
}
function centerTrim(tensor, reference) {
  const tensorLength = tensor.shape[tensor.shape.length - 1];
  const delta = tensorLength - reference;
  if (delta < 0) throw new Error("tensor must be larger than reference");
  if (delta === 0) return tensor;
  const start = Math.floor(delta / 2);
  const end = tensorLength - (delta - start);
  const shape = [...tensor.shape];
  const newLength = end - start;
  shape[shape.length - 1] = newLength;
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(totalSize);
  const outerSize = totalSize / newLength;
  for (let i = 0; i < outerSize; i++)
    for (let j = 0; j < newLength; j++)
      out[i * newLength + j] = tensor.data[i * tensorLength + start + j];
  return { data: out, shape };
}
async function applyInference(model, mix) {
  const length = mix.length;
  const validLength = model.validLength(length);
  const paddedMix = mix.padded(validLength);
  const trainingLength = Math.floor(model.segment * model.samplerate);
  const paddedPaddedMix = padToTrainingLength(paddedMix, trainingLength);
  const z = spec(paddedPaddedMix);
  const magspec = magnitude(z);
  const { outX, outXt } = await model.forward(paddedMix, magspec);
  const zout = maskToComplex(outX);
  const timeFromSpec = ispec(zout, trainingLength);
  const sumBeforeCrop = addTensors(outXt, timeFromSpec);
  const out = cropToValidLength(sumBeforeCrop, validLength);
  return centerTrim(out, length);
}
async function applySplits(model, mix, progressCallback, overlap = 0.25) {
  const [batch, channels, length] = mix.shape;
  const sources = model.sources.length;
  const outData = new Float32Array(batch * sources * channels * length);
  const sumWeight = new Float32Array(length);
  const segment = Math.floor(model.samplerate * model.segment);
  const stride = Math.floor((1 - overlap) * segment);
  const weight = new Float32Array(segment);
  for (let i = 0; i < Math.floor(segment / 2) + 1; i++) weight[i] = i + 1;
  for (let i = Math.floor(segment / 2) + 1; i < segment; i++) weight[i] = segment - i;
  const maxWeight = weight.reduce((mx, c) => Math.max(mx, c), -Infinity);
  for (let i = 0; i < segment; i++) weight[i] /= maxWeight;
  const total = Math.ceil(length / stride);
  if (progressCallback) progressCallback(0, total);
  let offset = 0, chunkIndex = 0;
  while (offset < length) {
    const chunk = new TensorChunk(mix, offset, segment);
    const chunkOut = await applyInference(model, chunk);
    const chunkLength = chunkOut.shape[chunkOut.shape.length - 1];
    for (let b = 0; b < batch; b++)
      for (let s = 0; s < sources; s++)
        for (let c = 0; c < channels; c++)
          for (let t = 0; t < chunkLength; t++) {
            const outIdx = b * sources * channels * length + s * channels * length + c * length + offset + t;
            const chunkIdx = b * sources * channels * chunkLength + s * channels * chunkLength + c * chunkLength + t;
            outData[outIdx] += weight[t] * chunkOut.data[chunkIdx];
          }
    for (let t = 0; t < chunkLength && offset + t < length; t++)
      sumWeight[offset + t] += weight[t];
    offset += stride;
    chunkIndex++;
    if (progressCallback) progressCallback(chunkIndex, total);
  }
  for (let i = 0; i < outData.length; i++) {
    const timeIdx = i % length;
    if (sumWeight[timeIdx] > 0) outData[i] /= sumWeight[timeIdx];
  }
  return { data: outData, shape: [batch, sources, channels, length] };
}
function planarize(channelData) {
  const channels = channelData.length;
  const samples = channelData[0].length;
  if (channels === 1) return channelData[0];
  const data = new Float32Array(channels * samples);
  for (let c = 0; c < channels; c++) data.set(channelData[c], c * samples);
  return data;
}
async function separateTracks(model, rawAudio, progressCallback, overlap = 0.25) {
  const { channelData, sampleRate } = rawAudio;
  const channels = channelData.length;
  const samples = channelData[0].length;
  const mix = { data: planarize(channelData), shape: [1, channels, samples] };
  const result = await applySplits(model, mix, progressCallback, overlap);
  const [, sources, , length] = result.shape;
  const tracks = {};
  for (let s = 0; s < sources; s++) {
    const stemChannels = [];
    for (let c = 0; c < channels; c++) {
      const startIdx = s * channels * length + c * length;
      stemChannels.push(new Float32Array(result.data.buffer, startIdx * 4, length));
    }
    tracks[model.sources[s]] = { channelData: stemChannels, sampleRate };
  }
  return tracks;
}

/* ---------------- model wrapper ---------------- */
class ONNXHTDemucs {
  constructor() {
    this.sources = ["drums", "bass", "other", "vocals"];
    this.audioChannels = 2;
    this.samplerate = 44100;
    this.segment = 7.8;
  }
  static async init(ortLib, modelWeights, providers) {
    const instance = new ONNXHTDemucs();
    instance.ort = ortLib;
    instance.session = await ortLib.InferenceSession.create(modelWeights, providers ? { executionProviders: providers } : undefined);
    instance.inputNames = instance.session.inputNames;
    instance.outputNames = instance.session.outputNames;
    return instance;
  }
  validLength(length) {
    const trainingLength = Math.floor(this.segment * this.samplerate);
    if (trainingLength < length) throw new Error("chunk longer than training length");
    return trainingLength;
  }
  async forward(mix, magspec) {
    const feeds = {};
    feeds[this.inputNames[0]] = new this.ort.Tensor("float32", mix.data, mix.shape);
    feeds[this.inputNames[1]] = new this.ort.Tensor("float32", magspec.data, magspec.shape);
    const results = await this.session.run(feeds);
    const outX = results[this.outputNames[0]];
    const outXt = results[this.outputNames[1]];
    return {
      outX: { data: outX.data, shape: outX.dims },
      outXt: { data: outXt.data, shape: outXt.dims },
    };
  }
}

/* ---------------- WAV encode (16-bit PCM) ---------------- */
function samplesToWav(channelData, sampleRate) {
  const channels = channelData.length;
  const samples = channelData[0].length;
  const dataSize = samples * channels * 2;
  const buffer = new Uint8Array(44 + dataSize);
  const view = new DataView(buffer.buffer);
  const writeString = (str, off) => { for (let i = 0; i < str.length; i++) buffer[off + i] = str.charCodeAt(i); };
  writeString("RIFF", 0); view.setUint32(4, 36 + dataSize, true); writeString("WAVE", 8);
  writeString("fmt ", 12); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, channels, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true); view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeString("data", 36); view.setUint32(40, dataSize, true);
  let off = 44;
  for (let s = 0; s < samples; s++)
    for (let c = 0; c < channels; c++) {
      const v = Math.max(-1, Math.min(1, channelData[c][s]));
      view.setInt16(off, Math.round(v * 32767), true);
      off += 2;
    }
  return buffer;
}

/* expose core for the Node validation harness */
if (typeof globalThis !== "undefined")
  globalThis.DemucsCore = { separateTracks, ONNXHTDemucs, samplesToWav };

/* ---------------- worker entry ---------------- */
if (typeof importScripts === "function") {
  const MODEL_PARTS = ["htdemucs.onnx.part1", "htdemucs.onnx.part2"];
  const MODEL_BYTES = 174263526;

  function openAiDb() {
    return new Promise((res, rej) => {
      const rq = indexedDB.open("karaoke-ai", 1);
      rq.onupgradeneeded = () => rq.result.createObjectStore("files");
      rq.onsuccess = () => res(rq.result);
      rq.onerror = () => rej(rq.error);
    });
  }
  async function cachedModel(base) {
    try {
      const db = await openAiDb();
      const hit = await new Promise((res) => {
        const rq = db.transaction("files").objectStore("files").get("htdemucs");
        rq.onsuccess = () => res(rq.result || null);
        rq.onerror = () => res(null);
      });
      if (hit && hit.size === MODEL_BYTES) return new Uint8Array(await hit.arrayBuffer());
    } catch (e) {}
    /* download parts with progress, then cache */
    const chunks = [];
    let loaded = 0;
    for (const part of MODEL_PARTS) {
      const resp = await fetch(base + part);
      if (!resp.ok) throw new Error("model download failed: " + resp.status);
      const reader = resp.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        postMessage({ type: "download", loaded, total: MODEL_BYTES });
      }
    }
    const full = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { full.set(c, off); off += c.byteLength; }
    if (full.byteLength !== MODEL_BYTES) throw new Error("model size mismatch: " + full.byteLength);
    try {
      const db = await openAiDb();
      const blob = new Blob([full], { type: "application/octet-stream" });
      await new Promise((res) => {
        const tx = db.transaction("files", "readwrite");
        tx.objectStore("files").put(blob, "htdemucs");
        tx.oncomplete = res; tx.onerror = res;
      });
    } catch (e) {}
    return full;
  }

  self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.cmd !== "separate") return;
    try {
      /* all assets live next to this worker script */
      const base = self.location.href.replace(/[^/?#]*([?#].*)?$/, "");
      importScripts(base + "ort/ort.all.min.js");
      ort.env.wasm.wasmPaths = base + "ort/";
      ort.env.wasm.numThreads = 1;         /* GitHub Pages has no COOP/COEP */
      const weights = await cachedModel(base);
      let model, backend = "webgpu";
      try {
        model = await ONNXHTDemucs.init(ort, weights, ["webgpu"]);
      } catch (err) {
        backend = "wasm";
        model = await ONNXHTDemucs.init(ort, weights, ["wasm"]);
      }
      /* normalize like the official demucs CLI (separate.py): zero-mean,
         unit-std of the mono reference — the model expects this range */
      const chs = msg.channelData;
      const nSamp = chs[0].length;
      let mean = 0;
      for (let i = 0; i < nSamp; i++) mean += (chs[0][i] + chs[1][i]) / 2;
      mean /= nSamp;
      let vs = 0;
      for (let i = 0; i < nSamp; i++) {
        const m = (chs[0][i] + chs[1][i]) / 2 - mean;
        vs += m * m;
      }
      const std = Math.sqrt(vs / nSamp) + 1e-8;
      for (const ch of chs)
        for (let i = 0; i < ch.length; i++) ch[i] = (ch[i] - mean) / std;

      const stems = await separateTracks(
        model,
        { channelData: chs, sampleRate: msg.sampleRate },
        (done, total) => postMessage({ type: "progress", done, total })
      );
      /* instrumental = mix - vocals (means cancel in the difference), scaled
         back to the original level; keeps every non-vocal detail intact */
      const nCh = chs.length;
      const inst = [];
      for (let c = 0; c < nCh; c++) {
        const mixCh = chs[c];
        const vocCh = stems.vocals.channelData[c];
        const out = new Float32Array(mixCh.length);
        for (let i = 0; i < mixCh.length; i++) out[i] = (mixCh[i] - vocCh[i]) * std;
        inst.push(out);
      }
      const wav = samplesToWav(inst, msg.sampleRate);
      postMessage({ type: "done", wav, backend }, [wav.buffer]);
    } catch (err) {
      postMessage({ type: "error", message: String(err && err.message || err) });
    }
  };
}

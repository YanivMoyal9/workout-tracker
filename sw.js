/* Service worker — offline support for the workout tracker.
   Network-first so the app always updates when online (cache-first would
   pin a stale version), falling back to the cache only when offline. */
var CACHE = "workout-tracker-v2";
var SHELL = [
  "./",
  "workout-tracker.html",
  "manifest.json"
];

self.addEventListener("install", function (e) {
  e.waitUntil(
    caches.open(CACHE).then(function (c) {
      // addAll fails the whole install if any request 404s, so add individually.
      return Promise.all(SHELL.map(function (url) {
        return c.add(url).catch(function () { /* ignore missing entry */ });
      }));
    }).then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener("activate", function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (k) {
        if (k !== CACHE) return caches.delete(k); // purge older caches
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener("fetch", function (e) {
  var req = e.request;
  if (req.method !== "GET") return;
  if (req.url.indexOf(self.location.origin) !== 0) return; // ignore cross-origin
  e.respondWith(
    fetch(req).then(function (res) {
      // Keep the cache fresh with the latest successful response.
      if (res && res.status === 200) {
        var copy = res.clone();
        caches.open(CACHE).then(function (c) { c.put(req, copy); });
      }
      return res;
    }).catch(function () {
      // Offline: serve from cache, falling back to the app shell for navigations.
      return caches.match(req).then(function (cached) {
        return cached || (req.mode === "navigate" ? caches.match("workout-tracker.html") : undefined);
      });
    })
  );
});

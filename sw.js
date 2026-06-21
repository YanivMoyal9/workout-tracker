/* Service worker — offline support for the workout tracker.
   Cache-first for the app shell so it works without a network in the gym. */
var CACHE = "workout-tracker-v1";
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
        if (k !== CACHE) return caches.delete(k);
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener("fetch", function (e) {
  var req = e.request;
  if (req.method !== "GET") return;
  e.respondWith(
    caches.match(req).then(function (cached) {
      if (cached) return cached;
      return fetch(req).then(function (res) {
        // Cache successful same-origin responses for future offline use.
        if (res && res.status === 200 && req.url.indexOf(self.location.origin) === 0) {
          var copy = res.clone();
          caches.open(CACHE).then(function (c) { c.put(req, copy); });
        }
        return res;
      }).catch(function () {
        // Offline and not cached: fall back to the app shell for navigations.
        if (req.mode === "navigate") return caches.match("workout-tracker.html");
      });
    })
  );
});

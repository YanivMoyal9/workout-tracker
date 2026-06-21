/* Kill-switch service worker.
   An earlier version of this app shipped a cache-first service worker that
   pinned a stale copy and served outdated code. This replacement takes over
   from that old worker, clears every cache, unregisters itself, and reloads
   open pages so clients return to the live network version. The app no longer
   registers a service worker at all. */
self.addEventListener("install", function () {
  self.skipWaiting();
});

self.addEventListener("activate", function (e) {
  e.waitUntil((async function () {
    try {
      var keys = await caches.keys();
      await Promise.all(keys.map(function (k) { return caches.delete(k); }));
    } catch (err) {}
    try { await self.registration.unregister(); } catch (err) {}
    try {
      var clients = await self.clients.matchAll({ type: "window" });
      clients.forEach(function (c) { if (typeof c.navigate === "function") c.navigate(c.url); });
    } catch (err) {}
  })());
});

/* Pass everything straight to the network — never serve from cache. */
self.addEventListener("fetch", function (e) {
  e.respondWith(fetch(e.request));
});

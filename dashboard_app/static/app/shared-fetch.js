const responseCache = new Map();

function toRequest(input, init = {}) {
  return input instanceof Request ? input : new Request(input, init);
}

function buildCacheKey(request) {
  if (request.method !== 'GET') return '';
  const url = new URL(request.url, window.location.origin);
  if (url.origin !== window.location.origin) return '';
  if (request.cache === 'no-store') return '';
  return `${request.method}:${url.href}`;
}

function shouldCacheResponse(response) {
  return response.ok;
}

function clearMatchingCache(predicate) {
  Array.from(responseCache.keys()).forEach((key) => {
    if (predicate(key)) responseCache.delete(key);
  });
}

function bustApiCacheFor(request, response) {
  if (!response.ok || request.method === 'GET') return;
  const url = new URL(request.url, window.location.origin);
  if (!url.pathname.startsWith('/api/internal/')) return;
  clearMatchingCache((key) => key.startsWith('GET:') && key.includes('/api/'));
}

export function clearSharedFetchCache() {
  responseCache.clear();
}

export async function sharedFetch(input, init = {}) {
  const request = toRequest(input, init);
  const cacheKey = buildCacheKey(request);

  if (cacheKey && responseCache.has(cacheKey)) {
    const cached = responseCache.get(cacheKey);
    const resolved = cached instanceof Promise ? await cached : cached;
    return resolved.clone();
  }

  const pending = (async () => {
    const response = await window.fetch(request);
    bustApiCacheFor(request, response);
    if (!shouldCacheResponse(response)) return response;
    return response.clone();
  })();

  if (cacheKey) responseCache.set(cacheKey, pending);

  try {
    const response = await pending;
    if (!cacheKey || !shouldCacheResponse(response)) {
      if (cacheKey) responseCache.delete(cacheKey);
      return response;
    }
    responseCache.set(cacheKey, response.clone());
    return response.clone();
  } catch (error) {
    if (cacheKey) responseCache.delete(cacheKey);
    throw error;
  }
}

if (typeof window !== 'undefined') {
  window.CrisisDashboardSharedFetch = sharedFetch;
  window.CrisisDashboardSharedFetchClear = clearSharedFetchCache;
}

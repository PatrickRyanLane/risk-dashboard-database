(function attachCrisisDashboardData(globalScope) {
  const root = globalScope || window;
  const csvCache = new Map();
  const jsonCache = new Map();
  const latestDateCache = new Map();

  function fetchTransport() {
    if (typeof root.CrisisDashboardSharedFetch === 'function') {
      return root.CrisisDashboardSharedFetch.bind(root);
    }
    return root.fetch.bind(root);
  }

  function normalizeUrl(url) {
    return new URL(url, root.location?.origin || window.location.origin).href;
  }

  function buildCacheKey(type, url, init = {}) {
    const method = String(init.method || 'GET').toUpperCase();
    if (method !== 'GET') return '';
    return `${type}:${normalizeUrl(url)}`;
  }

  async function memoized(cache, type, url, init, loader) {
    const cacheKey = buildCacheKey(type, url, init);
    if (!cacheKey) return loader();

    if (cache.has(cacheKey)) {
      const cached = cache.get(cacheKey);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = loader();
    cache.set(cacheKey, pending);

    try {
      const value = await pending;
      cache.set(cacheKey, value);
      return value;
    } catch (error) {
      cache.delete(cacheKey);
      throw error;
    }
  }

  function ensurePapa() {
    const papa = root.Papa || window.Papa;
    if (!papa) {
      throw new Error('PapaParse is not available.');
    }
    return papa;
  }

  async function request(url, init = {}) {
    const response = await fetchTransport()(url, { cache: 'default', ...init });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const payload = await response.json();
        if (payload?.error) detail = payload.error;
      } catch (_error) {
        // Ignore non-JSON error bodies.
      }
      throw new Error(detail);
    }
    return response;
  }

  async function fetchJson(url, init = {}) {
    return memoized(jsonCache, 'json', url, init, async () => {
      const response = await request(url, init);
      return response.json();
    });
  }

  async function fetchCsv(url, init = {}) {
    return memoized(csvCache, 'csv', url, init, async () => {
      const response = await request(url, init);
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        return response.json();
      }
      const text = await response.text();
      return new Promise((resolve, reject) => {
        ensurePapa().parse(text, {
          header: true,
          skipEmptyLines: true,
          complete: (output) => resolve(output.data || []),
          error: (error) => reject(error),
        });
      });
    });
  }

  async function fetchCsvAny(candidates, init = {}) {
    for (const url of candidates || []) {
      try {
        const rows = await fetchCsv(url, init);
        if (Array.isArray(rows)) return rows;
      } catch (_error) {
        // Keep trying until a candidate succeeds.
      }
    }
    return [];
  }

  async function fetchLatestDatedCsv({
    key,
    buildUrl,
    maxDays = 7,
    transform = (rows) => rows,
  }) {
    if (!key) throw new Error('A cache key is required.');
    if (typeof buildUrl !== 'function') throw new Error('buildUrl must be a function.');

    if (latestDateCache.has(key)) {
      const cachedDate = latestDateCache.get(key);
      const rows = await fetchCsv(buildUrl(cachedDate));
      return {
        data: transform(rows, cachedDate, 0),
        date: cachedDate,
        daysBack: 0,
      };
    }

    const now = new Date();
    const todayUtc = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    let lastError = null;

    for (let daysBack = 0; daysBack < maxDays; daysBack += 1) {
      const checkDate = new Date(todayUtc);
      checkDate.setUTCDate(checkDate.getUTCDate() - daysBack);
      const dateStr = checkDate.toISOString().split('T')[0];

      try {
        const rows = await fetchCsv(buildUrl(dateStr));
        latestDateCache.set(key, dateStr);
        return {
          data: transform(rows, dateStr, daysBack),
          date: dateStr,
          daysBack,
        };
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error(`No data found for ${key}`);
  }

  function invalidatePrefix(prefix) {
    [csvCache, jsonCache].forEach((cache) => {
      Array.from(cache.keys()).forEach((cacheKey) => {
        const [, url] = cacheKey.split(/:(.+)/);
        if (url && url.includes(prefix)) {
          cache.delete(cacheKey);
        }
      });
    });
    if (prefix.includes('/api/')) {
      latestDateCache.clear();
      root.CrisisDashboardSharedFetchClear?.();
    }
  }

  function clear() {
    csvCache.clear();
    jsonCache.clear();
    latestDateCache.clear();
    root.CrisisDashboardSharedFetchClear?.();
  }

  root.CrisisDashboardData = {
    fetchCsv,
    fetchCsvAny,
    fetchJson,
    fetchLatestDatedCsv,
    invalidatePrefix,
    clear,
  };
}(window));

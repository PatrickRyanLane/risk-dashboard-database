(function attachCrisisDashboardLegacy(globalScope) {
  const root = globalScope || window;

  function scheduleIdle(callback) {
    const idle = root.requestIdleCallback || ((cb) => root.setTimeout(cb, 0));
    return idle(callback);
  }

  function clearStatusLater(statusEl, delayMs = 3000) {
    if (!statusEl) return;
    root.setTimeout(() => {
      statusEl.textContent = '';
    }, delayMs);
  }

  async function runInternalRefresh({
    statusEl,
    refreshTargets = 'News, Organic SERP, SERP Features',
    refreshEndpoint = '/api/internal/refresh_aggregates',
    statusEndpoint = '/api/internal/refresh_aggregates/status',
  } = {}) {
    if (statusEl) statusEl.textContent = `Refreshing charts: ${refreshTargets}...`;
    try {
      const response = await root.fetch(refreshEndpoint, { method: 'POST' });
      if (!response.ok) throw new Error('refresh_failed');

      for (let i = 0; i < 240; i += 1) {
        const statusResponse = await root.fetch(statusEndpoint);
        if (!statusResponse.ok) throw new Error('status_failed');
        const payload = await statusResponse.json();
        if (payload.status !== 'in_progress') {
          if (statusEl) {
            if (payload.status === 'ok') {
              root.CrisisDashboardData?.invalidatePrefix?.('/api/');
              statusEl.textContent = `Refreshed charts: ${refreshTargets}`;
            } else if (payload.status === 'skipped') {
              statusEl.textContent = `Refresh skipped (locked): ${refreshTargets}`;
            } else if (payload.status === 'timeout') {
              statusEl.textContent = `Still refreshing: ${refreshTargets}`;
            } else {
              statusEl.textContent = `Refresh failed: ${refreshTargets}`;
            }
          }
          clearStatusLater(statusEl);
          return payload;
        }
        await new Promise((resolve) => root.setTimeout(resolve, 1000));
      }

      if (statusEl) statusEl.textContent = `Still refreshing: ${refreshTargets}`;
      clearStatusLater(statusEl);
      return { status: 'timeout' };
    } catch (_error) {
      if (statusEl) statusEl.textContent = `Refresh failed: ${refreshTargets}`;
      clearStatusLater(statusEl);
      return { status: 'error' };
    }
  }

  root.CrisisDashboardLegacy = {
    scheduleIdle,
    clearStatusLater,
    runInternalRefresh,
  };
}(window));

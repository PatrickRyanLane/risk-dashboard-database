const CHART_JS_SRC = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js';

let chartJsPromise = null;

function findScriptBySrc(src) {
  return Array.from(document.querySelectorAll('script[src]')).find((node) => {
    try {
      return new URL(node.src, window.location.origin).href === src;
    } catch (_error) {
      return false;
    }
  }) || null;
}

export async function ensureChartJs() {
  if (window.Chart) return window.Chart;
  if (chartJsPromise) return chartJsPromise;

  chartJsPromise = new Promise((resolve, reject) => {
    const existing = findScriptBySrc(CHART_JS_SRC);
    if (existing) {
      if (window.Chart) {
        resolve(window.Chart);
        return;
      }
      existing.addEventListener('load', () => resolve(window.Chart), { once: true });
      existing.addEventListener('error', () => reject(new Error('Failed to load Chart.js.')), { once: true });
      return;
    }

    const script = document.createElement('script');
    script.src = CHART_JS_SRC;
    script.async = true;
    script.addEventListener('load', () => resolve(window.Chart), { once: true });
    script.addEventListener('error', () => reject(new Error('Failed to load Chart.js.')), { once: true });
    document.head.appendChild(script);
  });

  return chartJsPromise;
}

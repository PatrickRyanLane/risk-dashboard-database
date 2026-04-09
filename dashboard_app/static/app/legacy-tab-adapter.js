import { sharedFetch } from './shared-fetch.js';

const pageCache = new Map();
const scriptCache = new Map();
const LEGACY_TAB_FILENAMES = new Set([
  'brand-dashboard.html',
  'ceo-dashboard.html',
  'sectors.html',
  'crises.html',
]);

function toAbsoluteUrl(value, base) {
  return new URL(value, base).href;
}

function stripScripts(doc) {
  doc.querySelectorAll('script').forEach((node) => node.remove());
}

function transformCss(cssText) {
  return String(cssText || '')
    .replace(/:root\b/g, ':host')
    .replace(/\bbody\b/g, ':host');
}

async function fetchPageDefinition(tabUrl) {
  const cacheKey = tabUrl.pathname;
  if (pageCache.has(cacheKey)) return pageCache.get(cacheKey);

  const loadPromise = (async () => {
    const response = await sharedFetch(cacheKey, { cache: 'default', credentials: 'same-origin' });
    if (!response.ok) {
      throw new Error(`Failed to load ${cacheKey}: HTTP ${response.status}`);
    }

    const html = await response.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    const scripts = Array.from(doc.querySelectorAll('script'));
    const externalScripts = [];
    const inlineScripts = [];

    scripts.forEach((script) => {
      if (script.src) {
        externalScripts.push(toAbsoluteUrl(script.getAttribute('src') || '', tabUrl.href));
      } else {
        inlineScripts.push(script.textContent || '');
      }
    });

    stripScripts(doc);

    const styles = Array.from(doc.querySelectorAll('style'))
      .map((node) => transformCss(node.textContent || ''))
      .join('\n\n');

    return {
      styles,
      markup: doc.body.innerHTML,
      externalScripts,
      inlineScript: inlineScripts.join('\n\n'),
    };
  })();

  pageCache.set(cacheKey, loadPromise);
  return loadPromise;
}

function getTabFilename(tabUrl) {
  const parts = String(tabUrl.pathname || '').split('/').filter(Boolean);
  return parts[parts.length - 1] || '';
}

function getLookbackDays(tabUrl, fallback = 30) {
  const days = Number(new URLSearchParams(tabUrl.search).get('days'));
  return Number.isFinite(days) && days > 0 ? days : fallback;
}

function buildLegacyPrewarmUrls(tabUrl) {
  const filename = getTabFilename(tabUrl);
  const days = getLookbackDays(tabUrl);

  if (filename === 'brand-dashboard.html') {
    return [
      `/api/v1/daily_counts?kind=brand_articles&days=${days}`,
      `/api/v1/daily_counts?kind=brand_serps&days=${days}`,
      `/api/v1/serp_features?entity=brand&days=${days}`,
      `/api/v1/serp_feature_controls?entity=brand&days=${days}`,
      '/api/v1/roster',
      '/api/dates',
    ];
  }
  if (filename === 'ceo-dashboard.html') {
    return [
      `/api/v1/daily_counts?kind=ceo_articles&days=${days}`,
      `/api/v1/daily_counts?kind=ceo_serps&days=${days}`,
      `/api/v1/serp_features?entity=ceo&days=${days}`,
      `/api/v1/serp_feature_controls?entity=ceo&days=${days}`,
      '/api/v1/roster',
      '/api/dates',
    ];
  }
  if (filename === 'sectors.html') {
    return [
      `/api/v1/daily_counts?kind=brand_articles&days=${days}`,
      `/api/v1/daily_counts?kind=brand_serps&days=${days}`,
      `/api/v1/serp_features?entity=brand&days=${days}`,
      `/api/v1/serp_feature_controls?entity=brand&days=${days}`,
      '/api/v1/roster',
      '/api/dates',
    ];
  }
  if (filename === 'crises.html') {
    return ['/api/dates'];
  }
  return [];
}

async function prewarmLegacyApiResponses(tabUrl) {
  const urls = buildLegacyPrewarmUrls(tabUrl);
  if (!urls.length) return;
  await Promise.allSettled(
    urls.map((url) => sharedFetch(url, { cache: 'default', credentials: 'same-origin' })),
  );
}

async function fetchBundledPageDefinition(tabUrl, shellConfig = {}) {
  if (!shellConfig?.useBundledLegacyPages) return null;
  const filename = getTabFilename(tabUrl);
  if (!LEGACY_TAB_FILENAMES.has(filename)) return null;
  const view = String(shellConfig.view || 'external').toLowerCase();
  if (view !== 'internal' && view !== 'external') return null;

  const cacheKey = `bundle:${view}:${filename}`;
  if (pageCache.has(cacheKey)) return pageCache.get(cacheKey);

  const loadPromise = (async () => {
    const modulePath = `/static/app/legacy-bundles/${view}/${filename.replace(/\.html$/i, '.js')}`;
    const loaded = await import(modulePath);
    const bundled = loaded?.default || loaded?.page || null;
    if (!bundled) {
      throw new Error(`Bundled legacy tab definition missing for ${view}/${filename}`);
    }
    return {
      styles: transformCss(bundled.styles || ''),
      markup: String(bundled.markup || ''),
      externalScripts: (Array.isArray(bundled.externalScripts) ? bundled.externalScripts : [])
        .map((src) => toAbsoluteUrl(String(src || ''), tabUrl.href)),
      inlineScript: String(bundled.inlineScript || ''),
    };
  })();

  pageCache.set(cacheKey, loadPromise);
  return loadPromise;
}

async function resolvePageDefinition(tabUrl, shellConfig = {}) {
  const bundled = await fetchBundledPageDefinition(tabUrl, shellConfig)
    .catch((error) => {
      console.warn(`Bundled legacy tab load failed for ${tabUrl.pathname}`, error);
      return null;
    });
  if (bundled) return bundled;
  return fetchPageDefinition(tabUrl);
}

export async function prefetchLegacyTab(tabUrl, {
  preloadScripts = false,
  prewarmApis = false,
  shellConfig = {},
} = {}) {
  const page = await resolvePageDefinition(tabUrl, shellConfig);
  if (preloadScripts) {
    await Promise.all(page.externalScripts.map((src) => ensureExternalScript(src)));
  }
  if (prewarmApis) {
    await prewarmLegacyApiResponses(tabUrl);
  }
  return page;
}

async function ensureExternalScript(src) {
  if (scriptCache.has(src)) return scriptCache.get(src);

  const loadPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-crisis-dashboard-src="${CSS.escape(src)}"]`);
    if (existing && existing.dataset.loaded === '1') {
      resolve();
      return;
    }
    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    script.async = false;
    script.dataset.crisisDashboardSrc = src;
    script.addEventListener('load', () => {
      script.dataset.loaded = '1';
      resolve();
    }, { once: true });
    script.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });
    document.head.appendChild(script);
  });

  scriptCache.set(src, loadPromise);
  return loadPromise;
}

function createEventTracker(cleanups) {
  return {
    add(target, type, listener, options) {
      target.addEventListener(type, listener, options);
      cleanups.push(() => target.removeEventListener(type, listener, options));
    },
  };
}

function createDocumentFacade(host, shadowRoot, tracker) {
  const getElementById = (id) => shadowRoot.getElementById
    ? shadowRoot.getElementById(id)
    : shadowRoot.querySelector(`#${CSS.escape(id)}`);

  return {
    getElementById,
    querySelector: shadowRoot.querySelector.bind(shadowRoot),
    querySelectorAll: shadowRoot.querySelectorAll.bind(shadowRoot),
    createElement: document.createElement.bind(document),
    createTextNode: document.createTextNode.bind(document),
    addEventListener(type, listener, options) {
      tracker.add(document, type, listener, options);
    },
    removeEventListener(type, listener, options) {
      document.removeEventListener(type, listener, options);
    },
    dispatchEvent: document.dispatchEvent.bind(document),
    body: host,
    documentElement: host,
    defaultView: window,
  };
}

function createLocationFacade(getDirectUrl, navigate) {
  const facade = {};
  Object.defineProperties(facade, {
    href: { get: () => getDirectUrl().href },
    origin: { get: () => getDirectUrl().origin },
    pathname: { get: () => getDirectUrl().pathname },
    search: { get: () => getDirectUrl().search },
    hash: { get: () => getDirectUrl().hash },
  });
  facade.assign = (next) => navigate(next, { replace: false });
  facade.replace = (next) => navigate(next, { replace: true });
  facade.toString = () => getDirectUrl().href;
  return facade;
}

function createWindowFacade(getDirectUrl, navigate, documentFacade, tracker) {
  const locationFacade = createLocationFacade(getDirectUrl, navigate);
  let clickHandler = null;

  const rawWindowFacade = {
    document: documentFacade,
    location: locationFacade,
    history: {
      replaceState(_state, _title, next) {
        navigate(next, { replace: true });
      },
      pushState(_state, _title, next) {
        navigate(next, { replace: false });
      },
    },
    addEventListener(type, listener, options) {
      tracker.add(window, type, listener, options);
    },
    removeEventListener(type, listener, options) {
      window.removeEventListener(type, listener, options);
    },
    dispatchEvent: window.dispatchEvent.bind(window),
    requestAnimationFrame: window.requestAnimationFrame.bind(window),
    cancelAnimationFrame: window.cancelAnimationFrame.bind(window),
    requestIdleCallback: window.requestIdleCallback
      ? window.requestIdleCallback.bind(window)
      : (cb) => window.setTimeout(() => cb({ didTimeout: false, timeRemaining: () => 0 }), 1),
    cancelIdleCallback: window.cancelIdleCallback
      ? window.cancelIdleCallback.bind(window)
      : window.clearTimeout.bind(window),
    setTimeout: window.setTimeout.bind(window),
    clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window),
    clearInterval: window.clearInterval.bind(window),
    getComputedStyle: window.getComputedStyle.bind(window),
    performance: window.performance,
    console: window.console,
    localStorage: window.localStorage,
    sessionStorage: window.sessionStorage,
    navigator: window.navigator,
  };

  Object.defineProperties(rawWindowFacade, {
    innerWidth: { get: () => window.innerWidth },
    innerHeight: { get: () => window.innerHeight },
  });

  Object.defineProperty(rawWindowFacade, 'onclick', {
    get() {
      return clickHandler;
    },
    set(handler) {
      if (clickHandler) {
        window.removeEventListener('click', clickHandler);
      }
      clickHandler = typeof handler === 'function' ? handler : null;
      if (clickHandler) {
        tracker.add(window, 'click', clickHandler);
      }
    },
  });

  const windowFacade = new Proxy(rawWindowFacade, {
    has() {
      return true;
    },
    get(target, prop) {
      if (prop in target) return target[prop];
      const value = window[prop];
      return value;
    },
    set(target, prop, value) {
      target[prop] = value;
      return true;
    },
  });

  rawWindowFacade.self = windowFacade;
  rawWindowFacade.top = windowFacade;
  rawWindowFacade.parent = windowFacade;
  return windowFacade;
}

function createScope(host, shadowRoot, getDirectUrl, navigate, cleanups) {
  const tracker = createEventTracker(cleanups);
  const documentFacade = createDocumentFacade(host, shadowRoot, tracker);
  const windowFacade = createWindowFacade(getDirectUrl, navigate, documentFacade, tracker);

  const baseScope = {
    window: windowFacade,
    self: windowFacade,
    top: windowFacade,
    parent: windowFacade,
    document: documentFacade,
    history: windowFacade.history,
    location: windowFacade.location,
    globalThis: windowFacade,
    fetch: sharedFetch,
    localStorage: window.localStorage,
    sessionStorage: window.sessionStorage,
    navigator: window.navigator,
    requestAnimationFrame: window.requestAnimationFrame.bind(window),
    cancelAnimationFrame: window.cancelAnimationFrame.bind(window),
    requestIdleCallback: windowFacade.requestIdleCallback,
    cancelIdleCallback: windowFacade.cancelIdleCallback,
    setTimeout: window.setTimeout.bind(window),
    clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window),
    clearInterval: window.clearInterval.bind(window),
    performance: window.performance,
    console: window.console,
    URL: window.URL,
    URLSearchParams: window.URLSearchParams,
    Chart: window.Chart,
    Papa: window.Papa,
    ResizeObserver: window.ResizeObserver,
    MutationObserver: window.MutationObserver,
    Event: window.Event,
    CustomEvent: window.CustomEvent,
    Node: window.Node,
    HTMLElement: window.HTMLElement,
    HTMLCanvasElement: window.HTMLCanvasElement,
    CSS: window.CSS,
    getComputedStyle: window.getComputedStyle.bind(window),
  };

  return new Proxy(baseScope, {
    has() {
      return true;
    },
    get(target, prop) {
      if (prop in target) return target[prop];
      const value = window[prop];
      return value;
    },
    set(target, prop, value) {
      target[prop] = value;
      return true;
    },
  });
}

function renderShadowMarkup(host, shadowRoot, page) {
  shadowRoot.innerHTML = `
    <style>
      :host{
        display:block;
        min-height:66vh;
      }
      ${page.styles}
    </style>
    ${page.markup}
  `;
  host.classList.add('shell-panel');
}

function executeInlineScript(page, scope, cacheKey) {
  if (!page.inlineScript.trim()) return;
  const runner = new Function(
    'scope',
    `with (scope) {\n${page.inlineScript}\n}\n//# sourceURL=${cacheKey.replace(/[^a-z0-9/_-]+/gi, '_')}.legacy.js`,
  );
  runner(scope);
}

export async function mountLegacyTab({ host, getDirectUrl, onHistoryChange, shellConfig = {} }) {
  const cleanups = [];
  const pageUrl = getDirectUrl();
  const page = await resolvePageDefinition(pageUrl, shellConfig);
  await Promise.all(page.externalScripts.map((src) => ensureExternalScript(src)));

  const shadowRoot = host.shadowRoot || host.attachShadow({ mode: 'open' });
  renderShadowMarkup(host, shadowRoot, page);

  const navigate = (next, { replace = true } = {}) => {
    onHistoryChange(next, { replace });
  };
  const scope = createScope(host, shadowRoot, getDirectUrl, navigate, cleanups);
  executeInlineScript(page, scope, pageUrl.pathname);
  window.dispatchEvent(new Event('resize'));

  return {
    show() {
      host.hidden = false;
      window.requestAnimationFrame(() => {
        window.dispatchEvent(new Event('resize'));
      });
    },
    hide() {
      host.hidden = true;
    },
    destroy() {
      while (cleanups.length) {
        const cleanup = cleanups.pop();
        try {
          cleanup();
        } catch (_err) {
          // Ignore listener cleanup errors during tab teardown.
        }
      }
      shadowRoot.innerHTML = '';
      host.classList.remove('shell-panel');
      host.hidden = true;
    },
  };
}

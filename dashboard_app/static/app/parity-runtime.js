import { sharedFetch } from './shared-fetch.js';

const scriptCache = new Map();
const modalScrollLock = {
  count: 0,
  bodyOverflow: '',
  htmlOverflow: '',
  bodyOverscroll: '',
  htmlOverscroll: '',
  bodyTouchAction: '',
};
const PARITY_THEME_OVERRIDES = `
  :host{
    --bg:#133238;
    --card:#0f3944;
    --muted:#cedcdd;
    --text:#ebf2f2;
    --accent:#58dbed;
    --black:#1f2121;
    --pill-neg:#ff8261;
    --pill-neu:#cedcdd;
    --pill-pos:#8be276;
    --stroke:rgba(94,133,142,.44);
    --chip:#0c2c35;
    --chipBorder:rgba(94,133,142,.52);
    color-scheme:dark;
    color:var(--text);
    font-family:"Proxima Nova", Roboto, "Segoe UI", Arial, sans-serif;
  }
  :host *{
    font-family:inherit;
  }
  :host .wrap{
    max-width:1440px !important;
    width:min(100%, 1440px) !important;
    padding-top:14px !important;
  }
  :host .card{
    background:transparent !important;
    border:0 !important;
    border-radius:0 !important;
    box-shadow:none !important;
    padding:0 !important;
  }
  :host .controls,
  :host .table-card,
  :host .chart-card,
  :host .refresh-panel{
    background:rgba(15,57,68,.58) !important;
    border:1px solid var(--stroke) !important;
    box-shadow:none !important;
    border-radius:10px !important;
  }
  :host .controls{
    padding:12px !important;
    gap:10px !important;
    margin:0 0 12px !important;
  }
  :host .controls .controls-inputs{
    display:grid !important;
    grid-template-columns:minmax(170px,1.05fr) minmax(230px,1.35fr) minmax(180px,1fr) minmax(180px,1fr) auto !important;
    gap:10px !important;
    align-items:end !important;
  }
  :host .controls .controls-inputs > *{
    min-width:0 !important;
  }
  :host .controls .controls-inputs > #clearBtn{
    grid-column:5 !important;
    grid-row:1 !important;
    justify-self:start !important;
    width:auto !important;
  }
  :host .controls .controls-inputs > #crisisBtn{
    grid-column:4 !important;
    grid-row:2 !important;
    justify-self:start !important;
    width:max-content !important;
    min-width:88px !important;
  }
  :host .controls .controls-inputs > #nonCrisisBtn{
    grid-column:5 !important;
    grid-row:2 !important;
    justify-self:start !important;
    width:max-content !important;
  }
  :host .controls .controls-inputs > #sectorFilterSelect{
    grid-column:3 !important;
    grid-row:1 !important;
  }
  :host .controls .controls-inputs > #companySizeSelect{
    grid-column:4 !important;
    grid-row:1 !important;
  }
  :host .chart-card{
    padding:12px !important;
  }
  :host .table-card{
    padding:10px !important;
  }
  :host .grid{
    gap:12px !important;
    margin:12px 0 !important;
  }
  :host .chart-card.chart-collapsed{
    border-radius:10px !important;
  }
  :host .chart-actions{
    top:12px !important;
    right:12px !important;
  }
  :host button,
  :host select,
  :host input[type="text"],
  :host input[type="search"]{
    border-radius:8px !important;
    border-color:rgba(94,133,142,.55) !important;
  }
  :host thead th,
  :host .field-label,
  :host .muted{
    color:var(--muted) !important;
  }
  :host .pagination{
    margin-top:8px !important;
  }
  :host tbody tr td:first-child{
    border-radius:8px 0 0 8px !important;
  }
  :host tbody tr td:last-child{
    border-radius:0 8px 8px 0 !important;
  }
  @media (max-width: 980px){
    :host .controls .controls-inputs{
      grid-template-columns:repeat(2, minmax(0, 1fr)) !important;
      gap:8px !important;
    }
    :host .controls .controls-inputs > .date-input-stack,
    :host .controls .controls-inputs > #filterInput,
    :host .controls .controls-inputs > #sectorFilterSelect,
    :host .controls .controls-inputs > #companySizeSelect,
    :host .controls .controls-inputs > #clearBtn{
      grid-column:1 / -1 !important;
      grid-row:auto !important;
      width:100% !important;
    }
    :host .controls .controls-inputs > #crisisBtn,
    :host .controls .controls-inputs > #nonCrisisBtn{
      grid-column:auto !important;
      grid-row:auto !important;
      width:100% !important;
      justify-self:stretch !important;
    }
  }
  @media (max-width: 620px){
    :host .controls{
      padding:10px !important;
      gap:8px !important;
    }
    :host .controls .controls-inputs{
      gap:7px !important;
    }
    :host .controls .lookback-controls{
      gap:6px !important;
    }
  }
  :host .modal.open{
    position:fixed !important;
    inset:0 !important;
    z-index:100000 !important;
    max-height:100dvh !important;
    overflow:auto !important;
  }
`;

function toAbsoluteUrl(value, base) {
  return new URL(value, base).href;
}

function transformCss(cssText) {
  return String(cssText || '')
    .replace(/:root\b/g, ':host')
    .replace(/\bbody\b/g, ':host');
}

export async function ensureExternalScript(src) {
  if (scriptCache.has(src)) return scriptCache.get(src);

  const loadPromise = new Promise((resolve, reject) => {
    const selector = `script[data-crisis-dashboard-src="${CSS.escape(src)}"]`;
    const existing = document.querySelector(selector);
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
      return window[prop];
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
      return window[prop];
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
      ${transformCss(page.styles)}
      ${PARITY_THEME_OVERRIDES}
    </style>
    ${page.markup}
  `;
  host.classList.add('shell-panel');
}

function lockPageScroll() {
  if (modalScrollLock.count === 0) {
    const bodyStyle = document.body?.style;
    const htmlStyle = document.documentElement?.style;
    modalScrollLock.bodyOverflow = bodyStyle?.overflow || '';
    modalScrollLock.htmlOverflow = htmlStyle?.overflow || '';
    modalScrollLock.bodyOverscroll = bodyStyle?.overscrollBehavior || '';
    modalScrollLock.htmlOverscroll = htmlStyle?.overscrollBehavior || '';
    modalScrollLock.bodyTouchAction = bodyStyle?.touchAction || '';
    if (bodyStyle) {
      bodyStyle.overflow = 'hidden';
      bodyStyle.overscrollBehavior = 'contain';
      bodyStyle.touchAction = 'none';
    }
    if (htmlStyle) {
      htmlStyle.overflow = 'hidden';
      htmlStyle.overscrollBehavior = 'contain';
    }
  }
  modalScrollLock.count += 1;
}

function unlockPageScroll() {
  if (modalScrollLock.count <= 0) return;
  modalScrollLock.count -= 1;
  if (modalScrollLock.count > 0) return;

  const bodyStyle = document.body?.style;
  const htmlStyle = document.documentElement?.style;
  if (bodyStyle) {
    bodyStyle.overflow = modalScrollLock.bodyOverflow;
    bodyStyle.overscrollBehavior = modalScrollLock.bodyOverscroll;
    bodyStyle.touchAction = modalScrollLock.bodyTouchAction;
  }
  if (htmlStyle) {
    htmlStyle.overflow = modalScrollLock.htmlOverflow;
    htmlStyle.overscrollBehavior = modalScrollLock.htmlOverscroll;
  }
}

function installModalViewportGuard(shadowRoot, cleanups) {
  let guardLocked = false;

  const sync = () => {
    const modals = Array.from(shadowRoot.querySelectorAll('.modal'));
    const openModals = modals.filter((modal) => modal.classList.contains('open'));

    const hasOpenModal = openModals.length > 0;
    if (hasOpenModal && !guardLocked) {
      lockPageScroll();
      guardLocked = true;
    } else if (!hasOpenModal && guardLocked) {
      unlockPageScroll();
      guardLocked = false;
    }
  };

  const observer = new MutationObserver(() => {
    sync();
  });
  observer.observe(shadowRoot, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ['class'],
  });
  cleanups.push(() => observer.disconnect());
  cleanups.push(() => {
    if (guardLocked) {
      unlockPageScroll();
      guardLocked = false;
    }
  });

  sync();
}

export async function prefetchParityPage({ page, getDirectUrl }) {
  const baseUrl = getDirectUrl();
  const scripts = (Array.isArray(page?.externalScripts) ? page.externalScripts : [])
    .map((src) => toAbsoluteUrl(String(src || ''), baseUrl.href));
  await Promise.all(scripts.map((src) => ensureExternalScript(src)));
}

export async function mountParityPage({
  host,
  getDirectUrl,
  onHistoryChange,
  page,
  runInlineScript,
}) {
  const cleanups = [];
  const pageUrl = getDirectUrl();
  const scripts = (Array.isArray(page?.externalScripts) ? page.externalScripts : [])
    .map((src) => toAbsoluteUrl(String(src || ''), pageUrl.href));
  await Promise.all(scripts.map((src) => ensureExternalScript(src)));

  const shadowRoot = host.shadowRoot || host.attachShadow({ mode: 'open' });
  renderShadowMarkup(host, shadowRoot, page);
  installModalViewportGuard(shadowRoot, cleanups);

  const navigate = (next, { replace = true } = {}) => {
    onHistoryChange(next, { replace });
  };
  const scope = createScope(host, shadowRoot, getDirectUrl, navigate, cleanups);
  runInlineScript(scope);
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

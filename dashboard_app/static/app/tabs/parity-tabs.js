const parityModuleCache = new Map();
const TAB_FILENAME = {
  brands: 'brand-dashboard.js',
  ceos: 'ceo-dashboard.js',
  sectors: 'sectors.js',
  crises: 'crises.js',
};
const TAB_HEADING = {
  brands: 'Brands',
  ceos: 'CEOs',
  sectors: 'Sector Overview',
  crises: 'Crises Overview',
};

function getParityModulePath(tabId, shellConfig = {}) {
  const filename = TAB_FILENAME[tabId];
  if (!filename) {
    throw new Error(`Unsupported parity tab: ${tabId}`);
  }
  const view = shellConfig?.view === 'internal' ? 'internal' : 'external';
  return `/static/app/parity-modules/${view}/${filename}`;
}

async function loadParityModule(tabId, shellConfig = {}) {
  const modulePath = getParityModulePath(tabId, shellConfig);
  if (parityModuleCache.has(modulePath)) {
    return parityModuleCache.get(modulePath);
  }
  const pending = import(modulePath);
  parityModuleCache.set(modulePath, pending);
  try {
    const loaded = await pending;
    parityModuleCache.set(modulePath, loaded);
    return loaded;
  } catch (error) {
    parityModuleCache.delete(modulePath);
    throw error;
  }
}

function applyParityHeading(host, tabId) {
  const headingText = TAB_HEADING[tabId];
  const shadowRoot = host?.shadowRoot;
  if (!headingText || !shadowRoot) return;
  const heading = shadowRoot.querySelector('.wrap h1, h1');
  if (!heading) return;
  heading.textContent = headingText;
}

export async function prefetchParityNativeTab({ tabId, shellConfig, getDirectUrl }) {
  const module = await loadParityModule(tabId, shellConfig);
  if (typeof module.prefetchParityTab !== 'function') {
    return;
  }
  await module.prefetchParityTab({ tabId, shellConfig, getDirectUrl });
}

export async function mountParityNativeTab({ tabId, shellConfig, host, getDirectUrl, onHistoryChange }) {
  const module = await loadParityModule(tabId, shellConfig);
  if (typeof module.mountParityTab !== 'function') {
    throw new Error(`Parity module is missing mountParityTab(): ${tabId}`);
  }
  const instance = await module.mountParityTab({
    tabId,
    shellConfig,
    host,
    getDirectUrl,
    onHistoryChange,
  });
  applyParityHeading(host, tabId);
  return {
    show() {
      applyParityHeading(host, tabId);
      instance?.show?.();
    },
    hide() {
      instance?.hide?.();
    },
    destroy() {
      instance?.destroy?.();
    },
  };
}

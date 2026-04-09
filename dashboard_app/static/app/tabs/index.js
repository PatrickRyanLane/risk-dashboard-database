import { mountCrisisTab, prefetchCrisisTab } from './crisis-tab.js';
import { mountEntityTab, prefetchEntityTab } from './entity-tab.js';
import { mountParityNativeTab, prefetchParityNativeTab } from './parity-tabs.js';
import { mountSectorTab, prefetchSectorTab } from './sector-tab.js';

const overviewNativeTabRegistry = {
  brands: {
    mount: mountEntityTab,
    prefetch: prefetchEntityTab,
  },
  ceos: {
    mount: mountEntityTab,
    prefetch: prefetchEntityTab,
  },
  sectors: {
    mount: mountSectorTab,
    prefetch: prefetchSectorTab,
  },
  crises: {
    mount: mountCrisisTab,
    prefetch: prefetchCrisisTab,
  },
};

const parityNativeTabRegistry = {
  brands: {
    mount: mountParityNativeTab,
    prefetch: prefetchParityNativeTab,
  },
  ceos: {
    mount: mountParityNativeTab,
    prefetch: prefetchParityNativeTab,
  },
  sectors: {
    mount: mountParityNativeTab,
    prefetch: prefetchParityNativeTab,
  },
  crises: {
    mount: mountParityNativeTab,
    prefetch: prefetchParityNativeTab,
  },
};

function getNativeTabRegistry(shellConfig = {}) {
  if (shellConfig?.nativeTabProfile === 'parity') {
    return parityNativeTabRegistry;
  }
  return overviewNativeTabRegistry;
}

export function isNativeTab(tabId) {
  return Object.prototype.hasOwnProperty.call(overviewNativeTabRegistry, tabId)
    || Object.prototype.hasOwnProperty.call(parityNativeTabRegistry, tabId);
}

export function prefetchNativeTab({ tabId, shellConfig, ...rest }) {
  const registry = getNativeTabRegistry(shellConfig);
  const entry = registry[tabId];
  if (!entry) return Promise.resolve();
  return entry.prefetch({ tabId, shellConfig, ...rest });
}

export function mountNativeTab({ tabId, shellConfig, ...rest }) {
  const registry = getNativeTabRegistry(shellConfig);
  const entry = registry[tabId];
  if (!entry) {
    throw new Error(`Unsupported native tab: ${tabId}`);
  }
  return entry.mount({ tabId, shellConfig, ...rest });
}

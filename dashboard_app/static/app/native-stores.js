import { clearAllEntityStores } from './entity-store.js';
import { clearAllSectorStores } from './sector-store.js';

export function clearAllNativeDataStores() {
  clearAllEntityStores();
  clearAllSectorStores();
}

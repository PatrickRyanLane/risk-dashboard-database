function getCardOrderStorage(storageKey) {
  try {
    const raw = localStorage.getItem(storageKey);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.map((value) => String(value || '').trim()).filter(Boolean) : [];
  } catch (_error) {
    return [];
  }
}

function setCardOrderStorage(storageKey, cardIds) {
  try {
    localStorage.setItem(storageKey, JSON.stringify(cardIds));
  } catch (_error) {
    // Ignore storage failures.
  }
}

function applyStoredOrder(groupEl, storageKey) {
  const orderedIds = getCardOrderStorage(storageKey);
  if (!orderedIds.length) return;
  const cards = Array.from(groupEl.querySelectorAll(':scope > .entity-card[data-card-id]'));
  if (!cards.length) return;
  const byId = new Map(cards.map((card) => [String(card.dataset.cardId || '').trim(), card]));
  orderedIds.forEach((cardId) => {
    const card = byId.get(cardId);
    if (card) groupEl.appendChild(card);
  });
}

function ensureDragHandle(card) {
  let handle = card.querySelector(':scope > .entity-card-drag-handle');
  if (handle) return handle;
  const legacyWrap = card.querySelector(':scope > .entity-card-drag-handle-wrap');
  if (legacyWrap) legacyWrap.remove();
  handle = document.createElement('button');
  handle.type = 'button';
  handle.className = 'entity-card-drag-handle';
  handle.setAttribute('aria-label', 'Drag to reorder card');
  handle.title = 'Drag to reorder card';
  card.appendChild(handle);
  return handle;
}

export function enableCardDragReorder({ root, storageNamespace = '', onReorder = null }) {
  if (!root) return [];
  const cleanups = [];
  const groups = Array.from(root.querySelectorAll('[data-card-group]'));

  groups.forEach((groupEl) => {
    const groupId = String(groupEl.getAttribute('data-card-group') || '').trim();
    if (!groupId) return;
    const cards = Array.from(groupEl.querySelectorAll(':scope > .entity-card[data-card-id]'));
    if (cards.length < 2) return;

    const storageKey = `riskdash.cardOrder:${storageNamespace || window.location.pathname}:${groupId}`;
    applyStoredOrder(groupEl, storageKey);

    const getCards = () => Array.from(groupEl.querySelectorAll(':scope > .entity-card[data-card-id]'));
    const saveOrder = () => {
      const ids = getCards()
        .map((card) => String(card.dataset.cardId || '').trim())
        .filter(Boolean);
      setCardOrderStorage(storageKey, ids);
    };

    let dragged = null;

    const clearDragClasses = () => {
      getCards().forEach((card) => {
        card.classList.remove('entity-card-drag-over');
        card.classList.remove('entity-card-dragging');
      });
    };

    getCards().forEach((card) => {
      card.dataset.draggable = 'true';
      card.removeAttribute('draggable');
      card.removeAttribute('title');
      const handle = ensureDragHandle(card);
      handle.setAttribute('draggable', 'true');

      const handleDragStart = (event) => {
        dragged = card;
        card.classList.add('entity-card-dragging');
        if (event.dataTransfer) {
          event.dataTransfer.effectAllowed = 'move';
          try {
            const rect = card.getBoundingClientRect();
            event.dataTransfer.setDragImage(card, Math.max(1, Math.round(rect.width / 2)), 12);
          } catch (_error) {
            // Ignore drag-image failures.
          }
          try {
            event.dataTransfer.setData('text/plain', String(card.dataset.cardId || ''));
          } catch (_error) {
            // Ignore transfer payload failures.
          }
        }
      };

      const handleDragOver = (event) => {
        event.preventDefault();
        if (!dragged || dragged === card) return;
        const rect = card.getBoundingClientRect();
        const before = event.clientY < (rect.top + rect.height / 2);
        groupEl.insertBefore(dragged, before ? card : card.nextSibling);
        card.classList.add('entity-card-drag-over');
      };

      const handleDragLeave = () => {
        card.classList.remove('entity-card-drag-over');
      };

      const handleDrop = (event) => {
        event.preventDefault();
        card.classList.remove('entity-card-drag-over');
      };

      const handleDragEnd = () => {
        clearDragClasses();
        dragged = null;
        saveOrder();
        if (typeof onReorder === 'function') onReorder();
      };

      handle.addEventListener('dragstart', handleDragStart);
      card.addEventListener('dragover', handleDragOver);
      card.addEventListener('dragleave', handleDragLeave);
      card.addEventListener('drop', handleDrop);
      handle.addEventListener('dragend', handleDragEnd);

      cleanups.push(() => {
        handle.removeEventListener('dragstart', handleDragStart);
        card.removeEventListener('dragover', handleDragOver);
        card.removeEventListener('dragleave', handleDragLeave);
        card.removeEventListener('drop', handleDrop);
        handle.removeEventListener('dragend', handleDragEnd);
      });
    });
  });

  return cleanups;
}

export async function runInternalRefresh(statusNode) {
  if (statusNode) statusNode.textContent = 'Refreshing aggregates...';
  const response = await fetch('/api/internal/refresh_aggregates', { method: 'POST' });
  if (!response.ok) {
    throw new Error('Refresh request failed.');
  }

  for (let attempt = 0; attempt < 240; attempt += 1) {
    const statusResponse = await fetch('/api/internal/refresh_aggregates/status');
    if (!statusResponse.ok) {
      throw new Error('Refresh status failed.');
    }
    const payload = await statusResponse.json();
    if (payload.status !== 'in_progress') {
      return payload.status;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }

  return 'timeout';
}

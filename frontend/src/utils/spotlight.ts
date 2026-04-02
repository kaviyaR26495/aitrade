/**
 * Spotlight utility — highlights a UI element by its data-guide-id attribute.
 * Used by the chatbot to visually guide users to specific controls.
 */

let cleanupFn: (() => void) | null = null;

export function spotlightElement(guideId: string, duration = 3500): void {
  // Clean up any previous spotlight
  dismissSpotlight();

  const target = document.querySelector(`[data-guide-id="${guideId}"]`) as HTMLElement | null;
  if (!target) return;

  // Scroll into view
  target.scrollIntoView({ behavior: 'smooth', block: 'center' });

  // Create overlay
  const overlay = document.createElement('div');
  overlay.className = 'spotlight-overlay';
  document.body.appendChild(overlay);

  // Add spotlight class to target
  target.classList.add('spotlight-target');

  // Click overlay to dismiss
  const handleClick = () => dismissSpotlight();
  overlay.addEventListener('click', handleClick);

  // Auto-dismiss after duration
  const timer = setTimeout(dismissSpotlight, duration);

  cleanupFn = () => {
    clearTimeout(timer);
    overlay.removeEventListener('click', handleClick);
    target.classList.remove('spotlight-target');
    overlay.remove();
    cleanupFn = null;
  };
}

export function dismissSpotlight(): void {
  cleanupFn?.();
}

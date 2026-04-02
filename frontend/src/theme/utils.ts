import { THEME_TOKENS } from './tokens';

/**
 * Helper: Get color based on status
 * Useful for displaying status badges, health indicators, etc.
 */
export const getColorByStatus = (status: 'success' | 'danger' | 'warning' | 'info'): string => {
  return THEME_TOKENS.colors.semantic[status].base;
};

/**
 * Helper: Get semantic color object with all variants
 */
export const getSemanticColor = (status: 'success' | 'danger' | 'warning' | 'info') => {
  return THEME_TOKENS.colors.semantic[status];
};

/**
 * Helper: Blend colors for hover/active states
 * Provides consistent interactive feedback
 */
export const blendColors = (color1: string, color2: string, ratio: number = 0.5): string => {
  // For CSS custom properties, return a calc expression
  // In a real app, you might use a color library for proper RGB blending
  return `color-mix(in srgb, ${color1} ${ratio * 100}%, ${color2})`;
};

/**
 * Helper: Get contrasting text color for a background
 */
export const getContrastText = (): string => {
  // Simple heuristic: dark backgrounds → light text, light backgrounds → dark text
  // In a real implementation, you'd calculate luminance properly
  return THEME_TOKENS.colors.text.primary;
};

/**
 * Helper: Create a color stop for gradients
 */
export const createGradientStop = (color: string, position: number): string => {
  return `${color} ${position}%`;
};

/**
 * Helper: Generate consistent spacing values
 */
export const getSpacing = (multiplier: number = 1): string => {
  return `${0.5 * multiplier}rem`;
};

/**
 * Helper: Get shadow with consistent apply
 */
export const applyShadow = (level: 'sm' | 'md' | 'lg' = 'md'): { boxShadow: string } => {
  return {
    boxShadow: THEME_TOKENS.shadows[level],
  };
};

/**
 * Helper: Get transition style
 */
export const getTransition = (property: string = 'all', speed: 'fast' | 'base' | 'slow' = 'base'): string => {
  const time = THEME_TOKENS.motion[speed];
  return `${property} ${time}`;
};

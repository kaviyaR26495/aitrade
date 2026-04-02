import { THEME_TOKENS } from './tokens';

/**
 * CSS Variables Generator
 * Converts theme tokens to CSS custom properties for use in stylesheets
 * Used in index.css to maintain parity between TS tokens and CSS
 */

export const createThemeVariables = (): { [key: string]: string } => {
  const vars: { [key: string]: string } = {};

  // Surface colors
  vars['--bg'] = THEME_TOKENS.colors.surfaces.page;
  vars['--bg-card'] = THEME_TOKENS.colors.surfaces.card;
  vars['--bg-card-solid'] = THEME_TOKENS.colors.surfaces.cardSolid;
  vars['--bg-input'] = THEME_TOKENS.colors.surfaces.input;
  vars['--bg-hover'] = THEME_TOKENS.colors.surfaces.hover;
  vars['--bg-active'] = THEME_TOKENS.colors.surfaces.active;
  vars['--bg-focus'] = THEME_TOKENS.colors.surfaces.focus;

  // Borders
  vars['--border'] = THEME_TOKENS.colors.borders.primary;
  vars['--border-light'] = THEME_TOKENS.colors.borders.light;
  vars['--border-focus'] = THEME_TOKENS.colors.borders.focus;

  // Text
  vars['--text'] = THEME_TOKENS.colors.text.primary;
  vars['--text-muted'] = THEME_TOKENS.colors.text.muted;
  vars['--text-dim'] = THEME_TOKENS.colors.text.dim;
  vars['--text-disabled'] = THEME_TOKENS.colors.text.disabled;

  // Brand
  vars['--primary'] = THEME_TOKENS.colors.brand.primary;
  vars['--primary-hover'] = THEME_TOKENS.colors.brand.primaryHover;
  vars['--primary-dark'] = THEME_TOKENS.colors.brand.primaryDark;
  vars['--primary-glow'] = THEME_TOKENS.colors.brand.glow;
  vars['--primary-subtle'] = THEME_TOKENS.colors.brand.subtle;

  // Semantic
  vars['--success'] = THEME_TOKENS.colors.semantic.success.base;
  vars['--success-glow'] = THEME_TOKENS.colors.semantic.success.glow;
  vars['--danger'] = THEME_TOKENS.colors.semantic.danger.base;
  vars['--danger-glow'] = THEME_TOKENS.colors.semantic.danger.glow;
  vars['--warning'] = THEME_TOKENS.colors.semantic.warning.base;
  vars['--warning-glow'] = THEME_TOKENS.colors.semantic.warning.glow;
  vars['--info'] = THEME_TOKENS.colors.semantic.info.base;
  vars['--info-glow'] = THEME_TOKENS.colors.semantic.info.glow;

  // Gradients
  vars['--gradient-start'] = '#6366f1';
  vars['--gradient-end'] = '#06b6d4';

  // Shadows
  vars['--shadow-sm'] = THEME_TOKENS.shadows.sm;
  vars['--shadow-md'] = THEME_TOKENS.shadows.md;
  vars['--shadow-lg'] = THEME_TOKENS.shadows.lg;
  vars['--shadow-glow'] = THEME_TOKENS.shadows.glow;

  // Layout
  vars['--content-max'] = THEME_TOKENS.layout.contentMax;
  vars['--sidebar-w'] = THEME_TOKENS.layout.sidebarWidth;
  vars['--sidebar-collapsed'] = THEME_TOKENS.layout.sidebarCollapsed;

  // Radius
  vars['--radius'] = THEME_TOKENS.radius.md;
  vars['--radius-lg'] = THEME_TOKENS.radius.lg;
  vars['--radius-sm'] = THEME_TOKENS.radius.sm;

  return vars;
};

/**
 * Injects theme CSS variables into the document root
 * Call this at app startup if not using CSS @theme block
 */
export const injectThemeVariables = (): void => {
  const vars = createThemeVariables();
  const root = document.documentElement;
  
  Object.entries(vars).forEach(([key, value]) => {
    root.style.setProperty(key, value);
  });
};

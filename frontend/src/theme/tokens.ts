/**
 * Design Tokens
 * Centralized color, spacing, typography, and shadow system
 * Used throughout the application for consistency
 */

export const THEME_TOKENS = {
  /* ─── Colors: Surfaces ─────────────────────────────────────────── */
  colors: {
    surfaces: {
      page: '#070c18',           // Darkest — page background
      card: '#0f1626',           // Card container — clearly distinct
      cardSolid: '#0f1626',
      input: '#0b1220',          // Form input — sunken into card
      hover: '#16213a',          // State: hover
      active: '#1a2845',         // State: active/selected
      focus: '#1e3050',          // State: focus
    },

    /* ─── Colors: Borders & Dividers ──────────────────────────────── */
    borders: {
      primary: '#1e3050',        // Main border — visible separator
      light: '#162440',          // Subtle divider, disabled element
      focus: '#6366f1',          // Focus ring
    },

    /* ─── Colors: Text & Typography ────────────────────────────────── */
    text: {
      primary: '#e2e8f4',        // Main text — high contrast
      muted: '#8894b0',          // Secondary text — lower emphasis
      dim: '#4a5878',            // Tertiary — very subtle
      disabled: '#3d4a62',       // Disabled state
    },

    /* ─── Colors: Brand ────────────────────────────────────────────── */
    brand: {
      primary: '#6366f1',        // Indigo — actions, highlights, focus
      primaryHover: '#818cf8',   // Lighter on hover
      primaryDark: '#4f46e5',    // Darker on press
      glow: 'rgba(99, 102, 241, 0.2)',      // Glow effect
      subtle: 'rgba(99, 102, 241, 0.1)',    // Subtle background
    },

    /* ─── Colors: Semantic – Status & Feedback ────────────────────── */
    semantic: {
      success: {
        base: '#10b981',
        glow: 'rgba(16, 185, 129, 0.15)',
        light: 'rgba(16, 185, 129, 0.1)',
      },
      danger: {
        base: '#f43f5e',
        glow: 'rgba(244, 63, 94, 0.15)',
        light: 'rgba(244, 63, 94, 0.1)',
      },
      warning: {
        base: '#f59e0b',
        glow: 'rgba(245, 158, 11, 0.15)',
        light: 'rgba(245, 158, 11, 0.1)',
      },
      info: {
        base: '#06b6d4',
        glow: 'rgba(6, 182, 212, 0.15)',
        light: 'rgba(6, 182, 212, 0.1)',
      },
    },

    /* ─── Colors: Gradients ────────────────────────────────────────── */
    gradients: {
      primary: 'linear-gradient(135deg, #6366f1, #06b6d4)',
      warm: 'linear-gradient(135deg, #f59e0b, #f43f5e)',
      cool: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
    },
  },

  /* ─── Shadows ──────────────────────────────────────────────────── */
  shadows: {
    sm: '0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3)',
    md: '0 4px 16px rgba(0,0,0,0.5)',
    lg: '0 12px 40px rgba(0,0,0,0.6)',
    glow: '0 0 32px rgba(99, 102, 241, 0.15)',
    inner: 'inset 0 1px 3px rgba(0,0,0,0.3)',
  },

  /* ─── Spacing Scale (8px base) ─────────────────────────────────── */
  spacing: {
    xs: '0.5rem',    // 8px
    sm: '1rem',      // 16px
    md: '1.5rem',    // 24px
    lg: '2rem',      // 32px
    xl: '3rem',      // 48px
    '2xl': '4rem',   // 64px
  },

  /* ─── Border Radius ────────────────────────────────────────────── */
  radius: {
    xs: '6px',
    sm: '8px',
    md: '12px',
    lg: '16px',
    xl: '20px',
    full: '9999px',
  },

  /* ─── Typography ──────────────────────────────────────────────── */
  typography: {
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    fontSize: {
      xs: '12px',
      sm: '13px',
      base: '14px',
      md: '15px',
      lg: '16px',
      xl: '18px',
      '2xl': '20px',
      '3xl': '24px',
      '4xl': '28px',
    },
    fontWeight: {
      light: 300,
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.75,
    },
    letterSpacing: '-0.01em',
  },

  /* ─── Transitions & Animations ────────────────────────────────── */
  motion: {
    fast: '0.15s ease',
    base: '0.2s ease',
    slow: '0.3s ease',
    easing: {
      in: 'ease-in',
      out: 'ease-out',
      inOut: 'ease-in-out',
    },
  },

  /* ─── Layout ──────────────────────────────────────────────────── */
  layout: {
    contentMax: '1440px',
    sidebarWidth: '260px',
    sidebarCollapsed: '72px',
  },

  /* ─── Z-Index Stack ────────────────────────────────────────────── */
  zIndex: {
    hide: -1,
    base: 0,
    dropdown: 10,
    sticky: 20,
    fixed: 30,
    modalBackdrop: 40,
    modal: 50,
    popover: 60,
    tooltip: 70,
    notification: 80,
  },
} as const;

export type ThemeTokens = typeof THEME_TOKENS;

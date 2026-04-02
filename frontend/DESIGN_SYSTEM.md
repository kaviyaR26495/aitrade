# AITrade Theme & Design System

## Overview

The AITrade theme system provides a **centralized, type-safe, professional design language** for the entire application. It ensures visual consistency, maintainability, and scalability across all pages and components.

### Key Features

✅ **Centralized Design Tokens** — Single source of truth for colors, typography, spacing, shadows  
✅ **Type-Safe** — Full TypeScript support with autocomplete  
✅ **Accessible** — High contrast, WCAG AA compliant  
✅ **Performant** — CSS variables for runtime efficiency  
✅ **Extensible** — Easy to customize and expand  
✅ **Documented** — Comprehensive guide and showcase

---

## Design Philosophy: Professional Dark Theme

### Color Palette

The app uses a **cohesive dark theme** with carefully chosen colors that establish visual hierarchy and depth.

#### Surface Hierarchy (Depth & Contrast)

```
Page Background      #070c18   ← Darkest (void/foundation)
  Card Surface       #0f1626   ← Clearly distinguished
    Input/Form       #0b1220   ← Sunken into card
      Hover State    #16213a   ← User feedback
        Active State #1a2845   ← Maximum interactivity
```

Each level is carefully spaced to provide **real visual distinction**. This is crucial for professional UI.

#### Borders & Dividers

- **Primary Border** (`#1e3050`): Visible, used for cards, inputs, section separators
- **Light Border** (`#162440`): Subtle dividers, disabled states, low-priority separators
- **Focus Ring** (`#6366f1`): Brand indigo, keyboard focus indicator

#### Text Hierarchy

```
Primary Text    #e2e8f4   ← Main content (high contrast)
Muted Text      #8894b0   ← Secondary, lower emphasis
Dim Text        #4a5878   ← Tertiary, very subtle labels
Disabled Text   #3d4a62   ← Non-interactive state
```

#### Brand Identity

- **Primary Color**: Indigo (`#6366f1`)
  - Use for: CTAs, highlights, focal points, brand elements
  - Hover: `#818cf8` (lighter)
  - Pressed: `#4f46e5` (darker)
  - Glow: `rgba(99, 102, 241, 0.2)` for depth effects

#### Semantic Status

Status information is communicated through dedicated colors:

| Status  | Color     | Use Case                                |
| ------- | --------- | --------------------------------------- |
| Success | `#10b981` | New data, positive actions, completions |
| Danger  | `#f43f5e` | Errors, destructive actions, stops      |
| Warning | `#f59e0b` | Alerts, cautions, needs attention       |
| Info    | `#06b6d4` | General information, neutral updates    |

Each includes variants:

- **Base**: Main color
- **Glow**: Light shadow effect (`rgba(color, 0.15)`)
- **Light**: Subtle background (`rgba(color, 0.1)`)

---

## File Structure

```
src/theme/
├── tokens.ts              # Design token definitions (SINGLE SOURCE OF TRUTH)
├── ThemeProvider.tsx      # React Context + useTheme() hook
├── cssVariables.ts        # CSS variable generation
├── utils.ts               # Helper functions (getColor, applyShadow, etc.)
├── index.ts               # Public exports
├── README.md              # Theme system documentation
└── [../components/]
    └── ThemeShowcase.tsx  # Interactive demo of all tokens
```

---

## Usage Guide

### 1. Access Tokens in React Components

Use the `useTheme()` hook to access design tokens:

```tsx
import { useTheme } from "@/theme";

export const MyCard = () => {
  const { tokens } = useTheme();

  return (
    <div
      style={{
        backgroundColor: tokens.colors.surfaces.card,
        color: tokens.colors.text.primary,
        padding: tokens.spacing.md,
        borderRadius: tokens.radius.lg,
        border: `1px solid ${tokens.colors.borders.primary}`,
      }}
    >
      Professional card with theme colors
    </div>
  );
};
```

### 2. Use CSS Variables (Preferred)

For better performance, use CSS variables directly in Tailwind classes:

```tsx
export const Card = ({ children }: { children: ReactNode }) => {
  return (
    <div className="bg-[var(--bg-card)] text-[var(--text)] rounded-[var(--radius-lg)] border border-[var(--border)] p-6 shadow-md">
      {children}
    </div>
  );
};
```

**Benefits:**

- Zero runtime overhead
- Works with Tailwind CSS
- Automatically synced with CSS @theme block
- Hot reload in development

### 3. Apply Semantic Colors

Use status colors for consistent feedback:

```tsx
import { getColorByStatus, getSemanticColor } from "@/theme";

export const StatusBadge = ({
  status,
}: {
  status: "success" | "danger" | "warning" | "info";
}) => {
  const { base, glow, light } = getSemanticColor(status);

  return (
    <span
      style={{
        backgroundColor: light,
        color: base,
        boxShadow: `0 0 12px ${glow}`,
        padding: "0.5rem 1rem",
        borderRadius: "999px",
        fontWeight: 600,
      }}
    >
      {status.toUpperCase()}
    </span>
  );
};
```

### 4. Use Typography Tokens

Maintain consistent text styling:

```tsx
import { useTheme } from "@/theme";

export const PageTitle = ({ children }: { children: string }) => {
  const { tokens } = useTheme();

  return (
    <h1
      style={{
        fontSize: tokens.typography.fontSize["3xl"],
        fontWeight: tokens.typography.fontWeight.bold,
        color: tokens.colors.text.primary,
        fontFamily: tokens.typography.fontFamily,
      }}
    >
      {children}
    </h1>
  );
};
```

### 5. Apply Transitions & Motion

Consistent, professional animations:

```tsx
import { getTransition } from '@/theme';

export const InteractiveButton = () => {
  return (
    <button style={{
      transition: getTransition('all', 'fast'),
      cursor: 'pointer,
    }}>
      Click me
    </button>
  );
};
```

---

## Component Patterns

### Card Component Pattern

```tsx
import { useTheme } from "@/theme";

export const Card = ({ title, children }) => {
  const { tokens } = useTheme();

  return (
    <div
      style={{
        backgroundColor: tokens.colors.surfaces.card,
        border: `1px solid ${tokens.colors.borders.primary}`,
        borderRadius: tokens.radius.lg,
        padding: tokens.spacing.lg,
        boxShadow: tokens.shadows.md,
      }}
    >
      {title && (
        <h3
          style={{
            fontSize: tokens.typography.fontSize.lg,
            fontWeight: tokens.typography.fontWeight.semibold,
            marginBottom: tokens.spacing.md,
          }}
        >
          {title}
        </h3>
      )}
      {children}
    </div>
  );
};
```

### Input Component Pattern

```tsx
export const Input = (props: React.InputHTMLAttributes<HTMLInputElement>) => {
  const { tokens } = useTheme();

  return (
    <input
      style={{
        backgroundColor: tokens.colors.surfaces.input,
        color: tokens.colors.text.primary,
        border: `1px solid ${tokens.colors.borders.primary}`,
        padding: `${tokens.spacing.sm} ${tokens.spacing.md}`,
        borderRadius: tokens.radius.md,
        fontSize: tokens.typography.fontSize.base,
        transition: getTransition("all", "base"),
      }}
      onFocus={(e) => {
        e.currentTarget.style.borderColor = tokens.colors.borders.focus;
        e.currentTarget.style.boxShadow = `0 0 0 3px rgba(99, 102, 241, 0.1)`;
      }}
      onBlur={(e) => {
        e.currentTarget.style.borderColor = tokens.colors.borders.primary;
        e.currentTarget.style.boxShadow = "none";
      }}
      {...props}
    />
  );
};
```

### Button Component Pattern

```tsx
export const Button = ({ variant = "primary", ...props }) => {
  const { tokens } = useTheme();

  const styles = {
    primary: {
      backgroundColor: tokens.colors.brand.primary,
      color: "#fff",
      border: "none",
    },
    secondary: {
      backgroundColor: "transparent",
      color: tokens.colors.brand.primary,
      border: `1px solid ${tokens.colors.brand.primary}`,
    },
  };

  return (
    <button
      style={{
        padding: `${tokens.spacing.sm} ${tokens.spacing.md}`,
        borderRadius: tokens.radius.md,
        fontSize: tokens.typography.fontSize.base,
        fontWeight: tokens.typography.fontWeight.semibold,
        cursor: "pointer",
        transition: getTransition("all", "fast"),
        ...styles[variant],
      }}
      {...props}
    />
  );
};
```

---

## Best Practices

### ✅ DO

- **Always use theme tokens** — Never hardcode colors
- **Prefer CSS variables** — Better performance than JavaScript lookups
- **Use semantic colors** — `success`, `danger`, `warning`, `info` for status
- **Maintain consistency** — Use spacing and radius consistently
- **Test for contrast** — Ensure text meets WCAG AA (minimum 4.5:1)
- **Document custom styling** — If breaking pattern, explain why
- **Use TypeScript** — Take advantage of autocomplete and type safety

### ❌ DON'T

- Hardcode color values: `backgroundColor: '#ff0000'` ❌
- Use random spacing: `padding: '17px'` ❌
- Mix token sources: CSS vars in one place, JS elsewhere ❌
- Ignore focus states: Keyboard navigation is essential ❌
- Over-engineer: Keep components simple and readable ❌

---

## Extending the Theme

### Adding a New Color

1. **Update `tokens.ts`:**

```ts
export const THEME_TOKENS = {
  colors: {
    brand: {
      primary: "#6366f1",
      secondary: "#YOUR_NEW_COLOR", // ← Add here
    },
  },
};
```

2. **Use in components:**

```tsx
const { tokens } = useTheme();
const color = tokens.colors.brand.secondary;
```

### Adding New Motion/Animation Speed

1. **Update `tokens.ts`:**

```ts
motion: {
  fast: '0.15s ease',
  verySlow: '0.5s ease-in-out', // ← Add here
}
```

2. **Use with helper:**

```ts
const transition = getTransition("opacity", "verySlow");
```

### Adding New Spacing Value

1. **Update `tokens.ts`:**

```ts
spacing: {
  xs: '0.5rem',
  custom: '1.25rem', // ← Add here
}
```

2. **Use in components:**

```tsx
padding: tokens.spacing.custom;
```

---

## Debugging & Troubleshooting

### Colors look wrong in the browser

**Issue:** Colors not matching expected values  
**Solution:**

1. Check `index.css` for @theme block — verify CSS vars match `tokens.ts`
2. Force refresh: Ctrl+Shift+R (clear cache)
3. Check browser DevTools → Elements → :root styles

### Build fails with TypeScript errors

**Issue:** `tokens is not defined` or similar  
**Solution:**

1. Ensure `ThemeProvider` wraps your app in `main.tsx`
2. Check `useTheme` is only called inside theme-wrapped components
3. Verify imports: `import { useTheme } from '@/theme'`

### Components not using theme

**Issue:** Old colors still showing  
**Solution:**

1. Replace hardcoded colors with CSS vars or `useTheme()`
2. Rebuild: `npm run dev` should hot-reload
3. Check component imports — ensure using updated version

---

## Show the Theme

To see all design tokens in action, import `ThemeShowcase`:

```tsx
import { ThemeShowcase } from "@/components/ThemeShowcase";

export default function App() {
  return <ThemeShowcase />;
}
```

This renders an interactive demo showing:

- All surface, border, text, and brand colors
- Semantic status colors
- Typography sizes and weights
- Spacing scale
- Border radius
- Shadow effects
- Live component examples (buttons, inputs, badges)

---

## Technical Integration

### React Context API

`ThemeProvider.tsx` uses React Context + `useTheme()` hook:

```tsx
const { tokens, isDark } = useTheme();
// tokens: Full THEME_TOKENS object
// isDark: true (fixed for this app)
```

### CSS Variables Sync

`cssVariables.ts` generates CSS custom properties from TypeScript tokens, ensuring no divergence between systems.

**Automatically synced:**

- `tokens.colors.brand.primary` ↔️ CSS `--primary`
- `tokens.colors.surfaces.card` ↔️ CSS `--bg-card`
- All tokens map to `--variable-names`

### TypeScript Support

Full type safety with autocomplete:

```tsx
const { tokens } = useTheme();
tokens.colors.surfaces.card; // ✓ Autocomplete works
tokens.colors.brand.notAColor; // ✗ TypeScript error
```

---

## Future Enhancements

- [ ] Light theme variant
- [ ] Theme customization panel (runtime color changes)
- [ ] Component library documentation (Storybook)
- [ ] Accessibility audit (WCAG AAA compliance)
- [ ] Animation performance optimization
- [ ] Figma design system plugin

---

## Design System Files

| File                | Purpose                                                  |
| ------------------- | -------------------------------------------------------- |
| `tokens.ts`         | Token definitions (colors, typography, spacing, shadows) |
| `ThemeProvider.tsx` | React Context + useTheme() hook                          |
| `cssVariables.ts`   | CSS variable generation & injection                      |
| `utils.ts`          | Helper functions (getColor, applyShadow, getTransition)  |
| `index.css`         | Global styles + @theme CSS registration                  |
| `ThemeShowcase.tsx` | Interactive demo of entire design system                 |

All components throughout the app automatically adapt to this centralized theme.

---

**Last Updated:** March 30, 2026  
**Status:** Production Ready  
**Theme Type:** Dark Mode Professional  
**Type Safety:** Full TypeScript

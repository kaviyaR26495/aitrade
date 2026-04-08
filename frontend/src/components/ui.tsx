import { type ReactNode, type HTMLAttributes, useEffect, useCallback, useRef, useState } from 'react';

/* ═══════════════════════════════════════════════════════════
   CARD — Primary content container
   ═══════════════════════════════════════════════════════════ */

interface CardProps extends Omit<HTMLAttributes<HTMLDivElement>, 'title'> {
  title?: ReactNode;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
  noPadding?: boolean;
  footer?: ReactNode;
}

export function Card({ title, children, className = '', action, noPadding, footer, ...rest }: CardProps) {
  return (
    <div
      className={`
        bg-[var(--bg-card)]/80 backdrop-blur-xl border border-[var(--border)] rounded-[var(--radius-lg)]
        shadow-[var(--shadow-sm)] transition-all duration-200
        hover:border-[color-mix(in_srgb,var(--border)_60%,var(--primary)_40%)]
        hover:shadow-[var(--shadow-md)]
        ${className}
      `}
      {...rest}
    >
      {(title || action) && (
        <div className="flex items-center justify-between gap-4 px-6 pt-5 lg:px-6 pb-4 border-b border-[var(--border-light)]">
          {title && (
            <h3 className="text-[12px] font-bold uppercase tracking-widest text-[var(--text-muted)]">{title}</h3>
          )}
          <div className="flex-shrink-0">{action}</div>
        </div>
      )}
      <div className={noPadding ? '' : 'px-6 py-5'}>
        {children}
      </div>
      {footer && (
        <div className="px-6 py-4 lg:px-7 border-t border-[var(--border-light)]">
          {footer}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   STAT CARD — Key metrics display
   ═══════════════════════════════════════════════════════════ */

interface StatCardProps extends HTMLAttributes<HTMLDivElement> {
  label: string;
  value: string | number;
  icon?: ReactNode;
  trend?: { value: number; label: string };
  color?: string;
  size?: 'sm' | 'md';
}

export function StatCard({ label, value, icon, trend, color = 'var(--primary)', size = 'md', ...rest }: StatCardProps) {
  const isSm = size === 'sm';
  return (
    <div
      className={`bg-[var(--bg-card)]/80 backdrop-blur-md border border-[var(--border)] rounded-[var(--radius-lg)]
                 transition-all duration-200 group relative overflow-hidden
                 hover:border-[color-mix(in_srgb,var(--border)_60%,var(--primary)_40%)]
                 hover:shadow-[var(--shadow-md)] ${isSm ? 'p-4' : 'p-5'}`}
      {...rest}
    >
      <div
        className="absolute -top-8 -right-8 w-24 h-24 rounded-full opacity-[0.07] blur-2xl
                   transition-opacity duration-500 group-hover:opacity-[0.15]"
        style={{ background: color }}
      />
      <div className="relative">
        <div className={`flex items-start justify-between ${isSm ? 'mb-2' : 'mb-3'}`}>
          <span className={`text-[var(--text-muted)] uppercase tracking-[0.1em] font-semibold leading-tight ${isSm ? 'text-[10px]' : 'text-[11px]'}`}>
            {label}
          </span>
          {icon && (
            <span
              className={`flex items-center justify-center rounded-[var(--radius-sm)] transition-colors ${isSm ? 'w-7 h-7' : 'w-9 h-9'}`}
              style={{ color, background: `color-mix(in srgb, ${color} 12%, transparent)` }}
            >
              {icon}
            </span>
          )}
        </div>
        <div
          className={`font-bold tracking-tight leading-none font-[var(--font-mono)] tabular-nums ${isSm ? 'text-xl' : 'text-[26px]'}`}
          style={{ color }}
        >
          {value}
        </div>
        {trend && (
          <div className={`flex items-center gap-1.5 text-xs mt-3 font-medium ${trend.value >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            <span className={`inline-flex items-center justify-center w-[18px] h-[18px] rounded-full text-[10px] ${
              trend.value >= 0 ? 'bg-emerald-400/10' : 'bg-rose-400/10'
            }`}>
              {trend.value >= 0 ? '↑' : '↓'}
            </span>
            <span className="tabular-nums">{Math.abs(trend.value)}%</span>
            <span className="text-[var(--text-dim)]">{trend.label}</span>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   BUTTON — Multi-variant action button
   ═══════════════════════════════════════════════════════════ */

interface ButtonProps extends Omit<HTMLAttributes<HTMLButtonElement>, 'type'> {
  children: ReactNode;
  onClick?: (e?: any) => void;
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  className?: string;
  type?: 'button' | 'submit';
  icon?: ReactNode;
  iconPosition?: 'left' | 'right';
}

export function Button({
  children, onClick, variant = 'primary', size = 'md',
  disabled, loading, className = '', type = 'button',
  icon, iconPosition = 'left', ...rest
}: ButtonProps) {
  const base = `
    inline-flex items-center justify-center font-semibold rounded-[var(--radius-sm)]
    transition-all duration-200 cursor-pointer select-none
    disabled:opacity-40 disabled:cursor-not-allowed disabled:pointer-events-none
    active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--primary)]/60
  `;

  const variants = {
    primary: `
      bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white
      shadow-sm hover:shadow-[0_0_20px_var(--primary-glow)]
    `,
    secondary: `
      bg-[var(--bg-hover)] hover:bg-[var(--bg-active)] text-[var(--text)]
      border border-[var(--border)] hover:border-[var(--text-muted)]
    `,
    danger: `
      bg-[var(--danger)] hover:brightness-110 text-white
      shadow-sm hover:shadow-[0_0_20px_var(--danger-glow)]
    `,
    ghost: `
      hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text)]
    `,
  };

  const sizes = {
    sm: 'h-9 min-w-[80px] px-4 text-[13px] gap-2 leading-none',
    md: 'h-10 min-w-[100px] px-5 text-sm gap-2 leading-none',
    lg: 'h-11 min-w-[120px] px-7 text-[15px] gap-2.5 leading-none',
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      {...rest}
    >
      {loading && (
        <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      )}
      {!loading && icon && iconPosition === 'left' && icon}
      {children}
      {!loading && icon && iconPosition === 'right' && icon}
    </button>
  );
}

/* ═══════════════════════════════════════════════════════════
   BADGE — Status indicators
   ═══════════════════════════════════════════════════════════ */

interface BadgeProps {
  children: ReactNode;
  color?: 'green' | 'red' | 'yellow' | 'blue' | 'gray' | 'purple';
  variant?: 'subtle' | 'solid' | 'outline';
  size?: 'sm' | 'md';
  dot?: boolean;
}

export function Badge({ children, color = 'blue', variant = 'subtle', size = 'sm', dot }: BadgeProps) {
  const colorMap = {
    green:  { subtle: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20', solid: 'bg-emerald-500 text-white border-emerald-500', outline: 'text-emerald-400 border-emerald-500/40', dot: 'bg-emerald-400' },
    red:    { subtle: 'bg-rose-500/10 text-rose-400 border-rose-500/20',       solid: 'bg-rose-500 text-white border-rose-500',       outline: 'text-rose-400 border-rose-500/40',    dot: 'bg-rose-400' },
    yellow: { subtle: 'bg-amber-500/10 text-amber-400 border-amber-500/20',     solid: 'bg-amber-500 text-white border-amber-500',     outline: 'text-amber-400 border-amber-500/40',  dot: 'bg-amber-400' },
    blue:   { subtle: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',   solid: 'bg-indigo-500 text-white border-indigo-500',   outline: 'text-indigo-400 border-indigo-500/40', dot: 'bg-indigo-400' },
    gray:   { subtle: 'bg-slate-500/10 text-slate-400 border-slate-500/20',     solid: 'bg-slate-500 text-white border-slate-500',     outline: 'text-slate-400 border-slate-500/40',  dot: 'bg-slate-400' },
    purple: { subtle: 'bg-violet-500/10 text-violet-400 border-violet-500/20',   solid: 'bg-violet-500 text-white border-violet-500',   outline: 'text-violet-400 border-violet-500/40', dot: 'bg-violet-400' },
  };
  const sizeClass = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs';
  const variantKey = variant === 'outline' ? 'outline' : variant;

  return (
    <span className={`
      inline-flex items-center gap-1.5 rounded-[5px]
      font-semibold border tracking-wide leading-none
      ${sizeClass} ${colorMap[color][variantKey]}
      ${variant === 'outline' ? 'bg-transparent' : ''}
    `}>
      {dot && <span className={`w-1.5 h-1.5 rounded-full ${colorMap[color].dot}`} />}
      {children}
    </span>
  );
}

/* ═══════════════════════════════════════════════════════════
   SELECT — Dropdown form control
   ═══════════════════════════════════════════════════════════ */

interface SelectProps extends Omit<HTMLAttributes<HTMLDivElement>, 'onChange'> {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  label?: string;
  className?: string;
}

export function Select({ value, onChange, options, label, className = '', ...rest }: SelectProps) {
  return (
    <div className={className} {...rest}>
      {label && <label className="block text-xs font-semibold text-[var(--text-muted)] mb-1.5 uppercase tracking-[0.05em]">{label}</label>}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="
          w-full h-10 bg-[var(--bg-input)] border border-[var(--border)]
          rounded-[var(--radius-sm)] px-3.5 text-sm text-[var(--text)]
          outline-none transition-all duration-200 cursor-pointer appearance-none
          hover:border-[var(--text-muted)]
          focus:border-[var(--primary)] focus:shadow-[0_0_0_3px_var(--primary-subtle)]
        "
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%237a8ba8' d='M6 8L1 3h10z'/%3E%3C/svg%3E")`,
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'right 10px center',
          paddingRight: '2rem',
        }}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   SEARCHABLE SELECT — Dropdown with search/filter
   ═══════════════════════════════════════════════════════════ */

interface SearchableSelectProps {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  label?: string;
  placeholder?: string;
  className?: string;
}

export function SearchableSelect({ value, onChange, options, label, placeholder = 'Search...', className = '' }: SearchableSelectProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [highlightIdx, setHighlightIdx] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const selectedLabel = options.find(o => o.value === value)?.label ?? '';

  const filtered = query
    ? options.filter(o => o.label.toLowerCase().includes(query.toLowerCase()))
    : options;

  useEffect(() => { setHighlightIdx(0); }, [query]);

  useEffect(() => {
    if (!open) { setQuery(''); }
  }, [open]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // scroll highlighted item into view
  useEffect(() => {
    if (!open || !listRef.current) return;
    const el = listRef.current.children[highlightIdx] as HTMLElement | undefined;
    el?.scrollIntoView({ block: 'nearest' });
  }, [highlightIdx, open]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!open) { if (e.key === 'ArrowDown' || e.key === 'Enter') { setOpen(true); e.preventDefault(); } return; }
    switch (e.key) {
      case 'ArrowDown': e.preventDefault(); setHighlightIdx(i => Math.min(i + 1, filtered.length - 1)); break;
      case 'ArrowUp':   e.preventDefault(); setHighlightIdx(i => Math.max(i - 1, 0)); break;
      case 'Enter':     e.preventDefault(); if (filtered[highlightIdx]) { onChange(filtered[highlightIdx].value); setOpen(false); } break;
      case 'Escape':    e.preventDefault(); setOpen(false); break;
    }
  };

  return (
    <div className={`relative ${className}`} ref={containerRef}>
      {label && <label className="block text-xs font-semibold text-[var(--text-muted)] mb-1.5 uppercase tracking-[0.05em]">{label}</label>}
      <button
        type="button"
        onClick={() => { setOpen(o => !o); setTimeout(() => inputRef.current?.focus(), 0); }}
        className="
          w-full h-10 bg-[var(--bg-input)] border border-[var(--border)]
          rounded-[var(--radius-sm)] px-3.5 text-sm text-left
          outline-none transition-all duration-200 cursor-pointer
          hover:border-[var(--text-muted)]
          focus:border-[var(--primary)] focus:shadow-[0_0_0_3px_var(--primary-subtle)]
          flex items-center justify-between
        "
      >
        <span className={selectedLabel ? 'text-[var(--text)]' : 'text-[var(--text-dim)]'}>
          {selectedLabel || placeholder}
        </span>
        <svg width="12" height="12" viewBox="0 0 12 12" className="flex-shrink-0 ml-2">
          <path fill="var(--text-dim)" d="M6 8L1 3h10z" />
        </svg>
      </button>

      {open && (
        <div className="
          absolute z-50 mt-1 w-full bg-[var(--bg-card)] border border-[var(--border)]
          rounded-[var(--radius-sm)] shadow-xl overflow-hidden
        ">
          <div className="p-2 border-b border-[var(--border)]">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              className="
                w-full h-8 bg-[var(--bg-input)] border border-[var(--border)]
                rounded-[var(--radius-sm)] px-2.5 text-sm text-[var(--text)]
                placeholder:text-[var(--text-dim)] outline-none
                focus:border-[var(--primary)]
              "
            />
          </div>
          <div ref={listRef} className="max-h-56 overflow-y-auto" onKeyDown={handleKeyDown}>
            {filtered.length === 0 ? (
              <div className="px-3 py-4 text-xs text-[var(--text-dim)] text-center">No results</div>
            ) : (
              filtered.map((o, i) => (
                <button
                  key={o.value}
                  type="button"
                  className={`
                    w-full text-left px-3 py-2 text-sm transition-colors cursor-pointer
                    ${i === highlightIdx ? 'bg-[var(--primary-subtle)] text-[var(--primary)]' : 'text-[var(--text)] hover:bg-[var(--bg-input)]'}
                    ${o.value === value ? 'font-medium' : ''}
                  `}
                  onMouseEnter={() => setHighlightIdx(i)}
                  onClick={() => { onChange(o.value); setOpen(false); }}
                >
                  {o.label}
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   INPUT — Text input field
   ═══════════════════════════════════════════════════════════ */

interface InputProps extends Omit<HTMLAttributes<HTMLDivElement>, 'onChange'> {
  value: string | number;
  onChange: (v: string) => void;
  label?: string;
  type?: string;
  placeholder?: string;
  className?: string;
}

export function Input({ value, onChange, label, type = 'text', placeholder, className = '', ...rest }: InputProps) {
  return (
    <div className={className} {...rest}>
      {label && <label className="block text-xs font-semibold text-[var(--text-muted)] mb-1.5 uppercase tracking-[0.05em]">{label}</label>}
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="
          w-full h-10 bg-[var(--bg-input)] border border-[var(--border)]
          rounded-[var(--radius-sm)] px-3.5 text-sm text-[var(--text)]
          placeholder:text-[var(--text-dim)] outline-none transition-all duration-200
          hover:border-[var(--text-muted)]
          focus:border-[var(--primary)] focus:shadow-[0_0_0_3px_var(--primary-subtle)]
        "
      />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   CHECKBOX — Styled checkbox input
   ═══════════════════════════════════════════════════════════ */

interface CheckboxProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  description?: string;
  className?: string;
}

export function Checkbox({ checked, onChange, label, description, className = '' }: CheckboxProps) {
  return (
    <label className={`flex items-start gap-3 cursor-pointer group ${className}`}>
      <span className="relative mt-0.5 flex-shrink-0">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="peer sr-only"
        />
        <span className={`
          flex items-center justify-center w-[18px] h-[18px]
          rounded-[4px] border transition-all duration-150
          ${checked
            ? 'bg-[var(--primary)] border-[var(--primary)]'
            : 'bg-[var(--bg-input)] border-[var(--border)] group-hover:border-[var(--text-dim)]'
          }
        `}>
          {checked && (
            <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
              <path d="M1 4L3.5 6.5L9 1" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </span>
      </span>
      {(label || description) && (
        <span>
          {label && <span className="text-sm font-medium text-[var(--text)]">{label}</span>}
          {description && <span className="block text-xs text-[var(--text-muted)] mt-0.5">{description}</span>}
        </span>
      )}
    </label>
  );
}

/* ═══════════════════════════════════════════════════════════
   TABLE — Data table with consistent styling
   ═══════════════════════════════════════════════════════════ */

export interface TableColumn<T = any> {
  key: string;
  label: string;
  align?: 'left' | 'center' | 'right';
  width?: string;
  render?: (row: T, index: number) => ReactNode;
  mono?: boolean;
}

interface TableProps<T = any> {
  columns: TableColumn<T>[];
  data: T[];
  onRowClick?: (row: T, index: number) => void;
  emptyState?: ReactNode;
  compact?: boolean;
  className?: string;
}

export function Table<T extends Record<string, any>>({ columns, data, onRowClick, emptyState, compact, className = '' }: TableProps<T>) {
  if (data.length === 0 && emptyState) {
    return <>{emptyState}</>;
  }

  const cellPad = compact ? 'py-2.5 px-3' : 'py-3 px-4';
  const alignClass = (a?: string) => a === 'center' ? 'text-center' : a === 'right' ? 'text-right' : 'text-left';

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--table-border)]">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`${cellPad} text-[11px] text-[var(--text-dim)] font-medium uppercase tracking-wider ${alignClass(col.align)}`}
                style={col.width ? { width: col.width } : undefined}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr
              key={i}
              onClick={onRowClick ? () => onRowClick(row, i) : undefined}
              className={`
                border-b border-[var(--table-border)]/30 hover:bg-[var(--table-row-hover)]
                transition-colors
                ${onRowClick ? 'cursor-pointer' : ''}
              `}
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={`${cellPad} ${alignClass(col.align)} ${col.mono ? 'font-[var(--font-mono)] tabular-nums' : ''}`}
                >
                  {col.render ? col.render(row, i) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   MODAL — Dialog overlay
   ═══════════════════════════════════════════════════════════ */

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  icon?: ReactNode;
  variant?: 'default' | 'danger';
  size?: 'sm' | 'md' | 'lg' | 'xl' | '2xl';
  children: ReactNode;
  footer?: ReactNode;
}

export function Modal({ open, onClose, title, description, icon, variant = 'default', size = 'md', children, footer }: ModalProps) {  // eslint-disable-line max-len
  const overlayRef = useRef<HTMLDivElement>(null);

  const handleEscape = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    if (open) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [open, handleEscape]);

  if (!open) return null;

  const sizeClass = size === '2xl' ? 'w-[min(1100px,95vw)]' : size === 'xl' ? 'w-[min(860px,95vw)]' : size === 'lg' ? 'w-[560px]' : size === 'sm' ? 'w-[360px]' : 'w-[440px]';
  const borderColor = variant === 'danger' ? 'border-rose-500/40' : 'border-[var(--modal-border)]';

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-[fade-in_0.15s_ease-out]"
      onClick={(e) => { if (e.target === overlayRef.current) onClose(); }}
    >
      <div className={`bg-[var(--modal-bg)] border ${borderColor} rounded-[var(--radius-xl)] p-7 ${sizeClass} max-h-[90vh] flex flex-col overflow-y-auto shadow-[var(--modal-shadow)] animate-[fade-in_0.2s_ease-out]`}>
        {(icon || title) && (
          <div className="flex items-center gap-3 mb-4">
            {icon && (
              <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
                variant === 'danger' ? 'bg-rose-500/15' : 'bg-[var(--primary-subtle)]'
              }`}>
                {icon}
              </div>
            )}
            <div>
              <h3 className={`text-lg font-bold ${variant === 'danger' ? 'text-rose-400' : 'text-[var(--text)]'}`}>{title}</h3>
              {description && <p className="text-sm text-[var(--text-muted)] mt-0.5">{description}</p>}
            </div>
          </div>
        )}
        {children}
        {footer && <div className="flex gap-3 mt-6">{footer}</div>}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   TABS — Tab navigation
   ═══════════════════════════════════════════════════════════ */

interface Tab {
  id: string;
  label: string;
  icon?: ReactNode;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (id: string) => void;
  variant?: 'underline' | 'pills';
  className?: string;
}

export function Tabs({ tabs, activeTab, onTabChange, variant = 'underline', className = '' }: TabsProps) {
  if (variant === 'pills') {
    return (
      <div className={`inline-flex gap-1 p-1 rounded-[var(--radius)] bg-[var(--bg-input)] border border-[var(--border)] ${className}`}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`
              inline-flex items-center gap-2 px-4 py-2 rounded-[var(--radius-sm)]
              text-sm font-medium transition-all duration-150 cursor-pointer
              ${activeTab === tab.id
                ? 'bg-[var(--primary)] text-white shadow-sm'
                : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-hover)]'
              }
            `}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className={`flex gap-0 border-b border-[var(--border)] ${className}`}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`
            inline-flex items-center gap-2 px-5 py-3 text-sm font-medium
            transition-all duration-150 relative cursor-pointer
            ${activeTab === tab.id
              ? 'text-[var(--primary)]'
              : 'text-[var(--text-muted)] hover:text-[var(--text)]'
            }
          `}
        >
          {tab.icon}
          {tab.label}
          {activeTab === tab.id && (
            <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-[var(--primary)] rounded-t-full" />
          )}
        </button>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   LIST ITEM — Consistent row component
   ═══════════════════════════════════════════════════════════ */

interface ListItemProps {
  left: ReactNode;
  right?: ReactNode;
  onClick?: () => void;
  active?: boolean;
  className?: string;
}

export function ListItem({ left, right, onClick, active, className = '' }: ListItemProps) {
  return (
    <div
      onClick={onClick}
      className={`
        flex items-center justify-between py-3 px-4 rounded-[var(--radius-sm)]
        text-sm transition-all duration-150
        ${onClick ? 'cursor-pointer' : ''}
        ${active
          ? 'bg-[var(--primary-subtle)] border border-[var(--primary)]/40 shadow-sm'
          : 'bg-[var(--bg-input)] hover:bg-[var(--bg-hover)]'
        }
        ${className}
      `}
    >
      <div className="flex items-center gap-2.5 min-w-0">{left}</div>
      {right && <div className="flex items-center gap-2 shrink-0">{right}</div>}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   SKELETON — Loading placeholders
   ═══════════════════════════════════════════════════════════ */

interface SkeletonProps {
  className?: string;
  lines?: number;
}

export function Skeleton({ className = 'h-10', lines }: SkeletonProps) {
  if (lines) {
    return (
      <div className="space-y-2">
        {Array.from({ length: lines }).map((_, i) => (
          <div key={i} className={`skeleton rounded-[var(--radius-sm)] ${i === lines - 1 ? 'w-3/4' : 'w-full'} ${className}`} />
        ))}
      </div>
    );
  }
  return <div className={`skeleton rounded-[var(--radius-sm)] ${className}`} />;
}

export function SkeletonTable({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      <Skeleton className="h-8 w-full" />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-12 w-full" />
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   TOOLTIP — Hover tooltip
   ═══════════════════════════════════════════════════════════ */

interface TooltipProps {
  content: string;
  side?: 'top' | 'right' | 'bottom' | 'left';
  children: ReactNode;
}

export function Tooltip({ content, side = 'top', children }: TooltipProps) {
  const [visible, setVisible] = useState(false);

  const pos = {
    top:    'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left:   'right-full top-1/2 -translate-y-1/2 mr-2',
    right:  'left-full top-1/2 -translate-y-1/2 ml-2',
  };

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <span className={`
          absolute ${pos[side]} px-2.5 py-1.5 rounded-[var(--radius-sm)]
          bg-[var(--bg-elevated)] border border-[var(--border)]
          text-xs text-[var(--text)] whitespace-nowrap z-50
          shadow-[var(--shadow-lg)] pointer-events-none
          animate-[fade-in_0.1s_ease-out]
        `}>
          {content}
        </span>
      )}
    </span>
  );
}

/* ═══════════════════════════════════════════════════════════
   EMPTY STATE — Placeholder for empty content
   ═══════════════════════════════════════════════════════════ */

interface EmptyStateProps {
  icon: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6">
      <div className="p-4 rounded-2xl bg-[var(--primary-subtle)] text-[var(--text-dim)] mb-4">
        {icon}
      </div>
      <h3 className="text-sm font-semibold text-[var(--text)] mb-1">{title}</h3>
      {description && (
        <p className="text-[13px] text-[var(--text-muted)] text-center max-w-sm leading-relaxed">{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   PAGE HEADER — Consistent page title section
   ═══════════════════════════════════════════════════════════ */

interface PageHeaderProps {
  title: string;
  description?: string;
  children?: ReactNode;
}

export function PageHeader({ title, description, children }: PageHeaderProps) {
  return (
    <div className="pb-6 mb-6 border-b border-[var(--border-light)]">
      <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-[22px] font-bold tracking-tight text-[var(--text)] leading-tight">{title}</h1>
          {description && (
            <p className="text-[13px] text-[var(--text-muted)] mt-1 leading-relaxed">{description}</p>
          )}
        </div>
        {children && <div className="flex flex-wrap items-center gap-2.5 xl:flex-shrink-0">{children}</div>}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   SECTION — Group related content with a label
   ═══════════════════════════════════════════════════════════ */

interface SectionProps {
  children: ReactNode;
  className?: string;
}

export function Section({ children, className = '' }: SectionProps) {
  return <div className={`space-y-5 ${className}`}>{children}</div>;
}

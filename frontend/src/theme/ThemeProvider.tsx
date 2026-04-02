import { useEffect } from 'react';
import type { ReactNode } from 'react';
import { useAppStore } from '../store/appStore';

export const ThemeProvider = ({ children }: { children: ReactNode }) => {
  const theme = useAppStore((s) => s.theme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  return <>{children}</>;
};

export const useTheme = () => {
  const theme = useAppStore((s) => s.theme);
  const setTheme = useAppStore((s) => s.setTheme);
  return { theme, setTheme, isDark: true };
};

export const withTheme = <P extends object>(
  Component: React.ComponentType<P & { theme: { theme: string; isDark: boolean } }>
): React.FC<P> => {
  return (props) => {
    const themeData = useTheme();
    return <Component {...(props as P)} theme={themeData} />;
  };
};
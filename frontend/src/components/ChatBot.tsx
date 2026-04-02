import { useState, useRef, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { MessageCircle, X, Send, Settings2, Trash2, ChevronDown } from 'lucide-react';
import { sendChatMessage, getChatProviders, getChatStatus, updateConfigBatch } from '../services/api';
import { spotlightElement } from '../utils/spotlight';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ProviderInfo {
  models: string[];
}

const ACTION_RE = /\[ACTION:(.*?)\]/g;

function parseActions(text: string): { cleanText: string; navigate?: string; highlight?: string } {
  let navigate: string | undefined;
  let highlight: string | undefined;

  const cleanText = text.replace(ACTION_RE, (_, params: string) => {
    for (const pair of params.split(',')) {
      const [k, v] = pair.split('=');
      if (k === 'navigate') navigate = v;
      if (k === 'highlight') highlight = v;
    }
    return '';
  }).trim();

  return { cleanText, navigate, highlight };
}

export default function ChatBot() {
  const location = useLocation();
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(() => localStorage.getItem('chatbot_open') === 'true');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [showConfig, setShowConfig] = useState(false);

  // Config state
  const [providers, setProviders] = useState<Record<string, ProviderInfo>>({});
  const [provider, setProvider] = useState('openai');
  const [model, setModel] = useState('gpt-4.1-nano');
  const [apiKey, setApiKey] = useState('');
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
  const [configLoaded, setConfigLoaded] = useState(false);
  const [configSaving, setConfigSaving] = useState(false);
  const [configured, setConfigured] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Persist open state
  useEffect(() => {
    localStorage.setItem('chatbot_open', String(isOpen));
  }, [isOpen]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load config on first open
  useEffect(() => {
    if (isOpen && !configLoaded) {
      loadConfig();
    }
  }, [isOpen, configLoaded]);

  const loadConfig = async () => {
    try {
      const [provRes, statusRes] = await Promise.all([getChatProviders(), getChatStatus()]);
      setProviders(provRes.data);
      setProvider(statusRes.data.provider || 'openai');
      setModel(statusRes.data.model || 'gpt-4.1-nano');
      setOllamaUrl(statusRes.data.ollama_base_url || 'http://localhost:11434');
      setConfigured(statusRes.data.configured);
      setConfigLoaded(true);
    } catch {
      setConfigLoaded(true);
    }
  };

  const handleSaveConfig = async () => {
    setConfigSaving(true);
    try {
      const settings = [
        { key: 'llm_provider', value: provider },
        { key: 'llm_model', value: model },
        { key: `llm_api_key_${provider}`, value: apiKey },
        { key: 'ollama_base_url', value: ollamaUrl },
      ];
      await updateConfigBatch(settings);
      setConfigured(true);
      setApiKey('');
      setShowConfig(false);
      await loadConfig();
    } catch {
      // silently fail
    } finally {
      setConfigSaving(false);
    }
  };

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || streaming) return;

    const userMsg: Message = { role: 'user', content: text };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setStreaming(true);

    // Add placeholder assistant message
    setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

    try {
      const reader = await sendChatMessage(
        newMessages.map((m) => ({ role: m.role, content: m.content })),
        location.pathname,
        provider,
        model,
      );

      const decoder = new TextDecoder();
      let fullContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          if (payload === '[DONE]') continue;

          try {
            const data = JSON.parse(payload);
            if (data.error) {
              fullContent += data.error;
            } else if (data.content) {
              fullContent += data.content;
            }
            // Update the last message in place
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: 'assistant', content: fullContent };
              return updated;
            });
          } catch {
            // skip unparseable
          }
        }
      }

      // Process actions from the complete response
      const { cleanText, navigate: navPath, highlight: guideId } = parseActions(fullContent);
      if (cleanText !== fullContent) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: cleanText };
          return updated;
        });
      }

      if (navPath && navPath !== location.pathname) {
        navigate(navPath);
        // Wait for page to render before spotlighting
        if (guideId) {
          setTimeout(() => spotlightElement(guideId), 400);
        }
      } else if (guideId) {
        setTimeout(() => spotlightElement(guideId), 100);
      }
    } catch {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: 'Failed to get a response. Check your LLM configuration in settings.',
        };
        return updated;
      });
    } finally {
      setStreaming(false);
    }
  }, [input, messages, streaming, location.pathname, provider, model, navigate]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const currentModels = providers[provider]?.models ?? [];

  // Render markdown-lite: bold, inline code, line breaks
  const renderContent = (text: string) => {
    return text.split('\n').map((line, i) => (
      <span key={i}>
        {i > 0 && <br />}
        {line.split(/(\*\*.*?\*\*|`[^`]+`)/).map((seg, j) => {
          if (seg.startsWith('**') && seg.endsWith('**'))
            return <strong key={j}>{seg.slice(2, -2)}</strong>;
          if (seg.startsWith('`') && seg.endsWith('`'))
            return <code key={j} className="bg-[var(--bg-input)] px-1.5 py-0.5 rounded-[var(--radius-sm)] text-xs font-mono text-[var(--primary)]">{seg.slice(1, -1)}</code>;
          return <span key={j}>{seg}</span>;
        })}
      </span>
    ));
  };

  return (
    <>
      {/* Floating button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 z-40 w-13 h-13 rounded-full bg-gradient-to-br from-[var(--primary)] to-indigo-700 text-white shadow-[var(--shadow-glow)] flex items-center justify-center transition-all hover:scale-110 hover:shadow-[0_0_24px_rgba(99,102,241,0.5)]"
          title="AITrade Assistant"
        >
          <MessageCircle size={22} />
        </button>
      )}

      {/* Chat panel */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 z-40 w-[400px] h-[550px] bg-[var(--bg-card-solid)] border border-[var(--border)] rounded-2xl shadow-[0_25px_60px_rgba(0,0,0,0.5)] flex flex-col overflow-hidden backdrop-blur-xl animate-[fade-in_0.2s_ease-out]">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-card-solid)]">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[var(--primary)] to-indigo-700 flex items-center justify-center">
                <MessageCircle size={14} className="text-white" />
              </div>
              <span className="font-semibold text-sm tracking-tight">AITrade Assistant</span>
              {configured && (
                <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]" title="LLM configured" />
              )}
            </div>
            <div className="flex items-center gap-0.5">
              <button onClick={clearChat} className="p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--bg-hover)] text-[var(--text-muted)] transition-colors" title="New Chat">
                <Trash2 size={14} />
              </button>
              <button onClick={() => setShowConfig(!showConfig)} className={`p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--bg-hover)] transition-colors ${showConfig ? 'text-[var(--primary)]' : 'text-[var(--text-muted)]'}`} title="Settings">
                <Settings2 size={14} />
              </button>
              <button onClick={() => setIsOpen(false)} className="p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--bg-hover)] text-[var(--text-muted)] transition-colors">
                <X size={14} />
              </button>
            </div>
          </div>

          {/* Config panel */}
          {showConfig && (
            <div className="px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-input)] space-y-3 text-xs">
              <div>
                <label className="block text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-medium mb-1.5">Provider</label>
                <select
                  value={provider}
                  onChange={(e) => {
                    setProvider(e.target.value);
                    const models = providers[e.target.value]?.models ?? [];
                    if (models.length > 0) setModel(models[0]);
                  }}
                  className="w-full bg-[var(--bg-card-solid)] border border-[var(--border)] rounded-[var(--radius-sm)] px-2.5 py-1.5 text-xs text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
                >
                  {Object.keys(providers).map((p) => (
                    <option key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-medium mb-1.5">Model</label>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full bg-[var(--bg-card-solid)] border border-[var(--border)] rounded-[var(--radius-sm)] px-2.5 py-1.5 text-xs text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)] mb-1.5"
                >
                  {currentModels.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
                <div className="flex flex-col gap-1.5">
                  <input
                    type="text"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder="Or type custom model name..."
                    className="w-full bg-[var(--bg-card-solid)] border border-[var(--border)] rounded-[var(--radius-sm)] px-2.5 py-1.5 text-xs text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
                  />
                  {provider === 'gemini' && (
                    <p className="text-[10px] text-[var(--text-muted)] italic leading-tight ml-1">
                      Tip: For simple chat navigation, <strong>gemini-1.5-flash</strong> or <strong>gemini-2.0-flash-lite</strong> are the most cost-effective and fastest options.
                    </p>
                  )}
                </div>
              </div>
              {provider !== 'ollama' ? (
                <div>
                  <label className="block text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-medium mb-1.5">API Key</label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder={configured ? '••••••• (saved)' : 'Enter API key'}
                    className="w-full bg-[var(--bg-card-solid)] border border-[var(--border)] rounded-[var(--radius-sm)] px-2.5 py-1.5 text-xs text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
                  />
                </div>
              ) : (
                <div>
                  <label className="block text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-medium mb-1.5">Ollama Base URL</label>
                  <input
                    type="text"
                    value={ollamaUrl}
                    onChange={(e) => setOllamaUrl(e.target.value)}
                    className="w-full bg-[var(--bg-card-solid)] border border-[var(--border)] rounded-[var(--radius-sm)] px-2.5 py-1.5 text-xs text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
                  />
                </div>
              )}
              <button
                onClick={handleSaveConfig}
                disabled={configSaving}
                className="w-full py-2 rounded-[var(--radius-sm)] bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white text-xs font-medium disabled:opacity-50 transition-colors"
              >
                {configSaving ? 'Saving...' : 'Save Configuration'}
              </button>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {messages.length === 0 && (
              <div className="text-center py-8">
                <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-[var(--primary-subtle)] flex items-center justify-center">
                  <MessageCircle size={24} className="text-[var(--primary)]" />
                </div>
                <p className="text-sm font-medium mb-1">Hi! I'm the AITrade Assistant.</p>
                <p className="text-xs text-[var(--text-muted)] leading-relaxed">Ask me anything about the app, or how to perform a task. I can even navigate you to the right page!</p>
                {!configured && (
                  <button
                    onClick={() => setShowConfig(true)}
                    className="mt-3 text-xs text-[var(--primary)] hover:underline font-medium"
                  >
                    Configure LLM provider to get started →
                  </button>
                )}
                <div className="mt-5 flex flex-wrap gap-2 justify-center">
                  {[
                    'How do I authenticate with Zerodha?',
                    'How to sync stock data?',
                    'What is regime classification?',
                    'How do I train a model?',
                  ].map((q) => (
                    <button
                      key={q}
                      onClick={() => { setInput(q); }}
                      className="text-[11px] px-3 py-1.5 rounded-full border border-[var(--border)] bg-[var(--bg-input)] text-[var(--text-muted)] hover:border-[var(--primary)] hover:text-[var(--primary)] hover:bg-[var(--primary-subtle)] transition-all"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-[85%] px-3.5 py-2.5 text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-br from-[var(--primary)] to-indigo-700 text-white rounded-2xl rounded-br-md shadow-[var(--shadow-sm)]'
                      : 'bg-[var(--bg-input)] text-[var(--text)] rounded-2xl rounded-bl-md border border-[var(--border)]'
                  }`}
                >
                  {msg.role === 'assistant' ? renderContent(msg.content) : msg.content}
                  {msg.role === 'assistant' && streaming && i === messages.length - 1 && (
                    <span className="inline-block w-1.5 h-4 ml-0.5 bg-[var(--primary)] animate-pulse rounded-sm" />
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="px-4 py-3 border-t border-[var(--border)]">
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about AITrade..."
                disabled={streaming}
                className="flex-1 bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius)] px-3 py-2 text-sm text-[var(--text)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/40 focus:border-[var(--primary)] disabled:opacity-50 transition-all"
              />
              <button
                onClick={handleSend}
                disabled={streaming || !input.trim()}
                className="px-3 py-2 bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white rounded-[var(--radius)] disabled:opacity-30 transition-all hover:shadow-[var(--shadow-glow)]"
              >
                <Send size={16} />
              </button>
            </div>
            <div className="flex items-center justify-between mt-1.5">
              <span className="text-[10px] text-[var(--text-dim)] font-mono">
                {provider}/{model}
              </span>
              <span className="text-[10px] text-[var(--text-dim)]">
                {location.pathname}
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

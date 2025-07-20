import React, { useState, useEffect } from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';

interface APIProviderToggleProps {
  className?: string;
}

export function APIProviderToggle({ className = '' }: APIProviderToggleProps) {
  const [currentProvider, setCurrentProvider] = useState<string>('openai');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch provider information on component mount
  useEffect(() => {
    fetchProviderInfo();
  }, []);

  const fetchProviderInfo = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch('/api/provider/info');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setCurrentProvider(data.current_provider || 'openai');

    } catch (err) {
      console.error('Failed to fetch provider info:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      setCurrentProvider('openai');
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggle = async () => {
    const newProvider = currentProvider === 'openai' ? 'kimi' : 'openai';
    
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch('/api/provider/switch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ provider: newProvider }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        setCurrentProvider(newProvider);
      } else {
        throw new Error(data.message || 'Failed to switch provider');
      }

    } catch (err) {
      console.error('Failed to switch provider:', err);
      setError(err instanceof Error ? err.message : 'Failed to switch provider');
    } finally {
      setIsLoading(false);
    }
  };

  const getProviderDisplayName = (provider: string): string => {
    const names: Record<string, string> = {
      'openai': 'OpenAI',
      'kimi': 'Kimi K2',
    };
    return names[provider] || provider.charAt(0).toUpperCase() + provider.slice(1);
  };

  const getProviderIcon = (provider: string): string => {
    const icons: Record<string, string> = {
      'openai': '🤖',
      'kimi': '🌙',
    };
    return icons[provider] || '🔧';
  };

  return (
    <div className={`flex items-center ${className}`}>
      {/* Simple Toggle Button */}
      <button
        onClick={handleToggle}
        disabled={isLoading}
        className={`
          flex items-center space-x-2 px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200
          ${currentProvider === 'openai' 
            ? 'bg-blue-600 hover:bg-blue-700 text-white' 
            : 'bg-purple-600 hover:bg-purple-700 text-white'
          }
          disabled:opacity-50 disabled:cursor-not-allowed
          shadow-sm hover:shadow-md
        `}
        title={`Switch to ${currentProvider === 'openai' ? 'Kimi K2' : 'OpenAI'}`}
      >
        {isLoading ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <span className="text-base">{getProviderIcon(currentProvider)}</span>
        )}
        
        <span className="font-semibold">
          {getProviderDisplayName(currentProvider)}
        </span>
      </button>

      {/* Error indicator */}
      {error && (
        <div className="ml-2" title={error}>
          <AlertCircle className="w-4 h-4 text-red-500" />
        </div>
      )}
    </div>
  );
}

export default APIProviderToggle;
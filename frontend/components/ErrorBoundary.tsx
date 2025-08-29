'use client'

import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertTriangle } from 'lucide-react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    
    // Log specific React rendering errors
    if (error.message.includes('Objects are not valid as a React child')) {
      console.error('🚨 OBJECT LITERAL DETECTED IN REACT:', error.message)
      console.error('This usually means object literals like {\'type\': \'paragraph\'} are in the content')
    }
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            <span className="font-medium">Content Rendering Error</span>
          </div>
          <p className="mt-1 text-sm">
            The message content contains formatting that couldn't be displayed.
          </p>
          {this.state.error && (
            <details className="mt-2 text-xs">
              <summary className="cursor-pointer font-mono">Technical Details</summary>
              <pre className="mt-1 whitespace-pre-wrap break-words">
                {this.state.error.message}
              </pre>
            </details>
          )}
        </div>
      )
    }

    return this.props.children
  }
}
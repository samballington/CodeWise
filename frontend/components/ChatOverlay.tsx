import React from 'react';
import ChatInterface from './ChatInterface';

interface ChatOverlayProps {
  onClose: () => void;
}

const ChatOverlay: React.FC<ChatOverlayProps> = ({ onClose }) => {
  return (
    <div className="fixed bottom-4 right-4 w-96 h-[28rem] bg-white shadow-lg border border-gray-200 rounded-lg flex flex-col z-50">
      {/* Header */}
      <div className="h-10 flex items-center justify-between px-3 border-b border-gray-200 bg-gray-50 rounded-t-lg">
        <span className="text-sm font-medium text-gray-800">Chat</span>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 focus:outline-none"
          title="Close chat"
        >
          ✕
        </button>
      </div>
      {/* Chat Interface */}
      <div className="flex-1 overflow-hidden">
        <ChatInterface />
      </div>
    </div>
  );
};

export default ChatOverlay; 
#!/usr/bin/env python3
"""
Voice Chat UI Component Generator
Creates React components for Turkish LLM voice interface
"""

import os
import json
from pathlib import Path


def create_react_components():
    """Create React components for voice chat interface."""

    # Create UI directory structure
    ui_dir = Path("ui")
    ui_dir.mkdir(exist_ok=True)

    # Create package.json
    package_json = {
        "name": "turkish-llm-voice-ui",
        "version": "1.0.0",
        "description": "Voice Chat Interface for Turkish LLM",
        "main": "index.js",
        "scripts": {"dev": "vite", "build": "vite build", "preview": "vite preview"},
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "lucide-react": "^0.263.1",
            "framer-motion": "^10.16.4",
            "clsx": "^2.0.0",
            "tailwind-merge": "^1.14.0",
        },
        "devDependencies": {
            "@types/react": "^18.2.15",
            "@types/react-dom": "^18.2.7",
            "@vitejs/plugin-react": "^4.0.3",
            "autoprefixer": "^10.4.14",
            "postcss": "^8.4.27",
            "tailwindcss": "^3.3.3",
            "typescript": "^5.0.2",
            "vite": "^4.4.5",
        },
    }

    with open(ui_dir / "package.json", "w") as f:
        json.dump(package_json, f, indent=2)

    # Create Vite config
    vite_config = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': 'ws://localhost:8765'
    }
  }
})
"""

    with open(ui_dir / "vite.config.ts", "w") as f:
        f.write(vite_config)

    # Create Tailwind config
    tailwind_config = """/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        accent: {
          50: '#f0fdf4',
          500: '#22c55e',
          600: '#16a34a',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
      }
    },
  },
  plugins: [],
}
"""

    with open(ui_dir / "tailwind.config.js", "w") as f:
        f.write(tailwind_config)

    # Create PostCSS config
    postcss_config = """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""

    with open(ui_dir / "postcss.config.js", "w") as f:
        f.write(postcss_config)

    # Create src directory
    src_dir = ui_dir / "src"
    src_dir.mkdir(exist_ok=True)

    # Create main App component
    app_component = """import React, { useState, useEffect } from 'react';
import { VoiceChat } from './components/VoiceChat';
import { SectorSelector } from './components/SectorSelector';
import { ChatHistory } from './components/ChatHistory';
import { Header } from './components/Header';

function App() {
  const [selectedSector, setSelectedSector] = useState('general');
  const [chatHistory, setChatHistory] = useState([]);

  const addToHistory = (message) => {
    setChatHistory(prev => [...prev, message]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Sidebar - Sector Selection */}
          <div className="lg:col-span-1">
            <SectorSelector 
              selectedSector={selectedSector}
              onSectorChange={setSelectedSector}
            />
          </div>
          
          {/* Main Content - Voice Chat */}
          <div className="lg:col-span-2">
            <VoiceChat 
              sector={selectedSector}
              onMessage={addToHistory}
            />
          </div>
        </div>
        
        {/* Chat History */}
        <div className="mt-8">
          <ChatHistory messages={chatHistory} />
        </div>
      </main>
    </div>
  );
}

export default App;
"""

    with open(src_dir / "App.tsx", "w") as f:
        f.write(app_component)

    # Create main.tsx
    main_tsx = """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
"""

    with open(src_dir / "main.tsx", "w") as f:
        f.write(main_tsx)

    # Create index.css
    index_css = """@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  }
}

@layer components {
  .btn-primary {
    @apply bg-primary-600 hover:bg-primary-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200;
  }
  
  .btn-secondary {
    @apply bg-gray-200 hover:bg-gray-300 text-gray-800 font-medium py-2 px-4 rounded-lg transition-colors duration-200;
  }
  
  .card {
    @apply bg-white rounded-xl shadow-lg border border-gray-100 p-6;
  }
}
"""

    with open(src_dir / "index.css", "w") as f:
        f.write(index_css)

    # Create components directory
    components_dir = src_dir / "components"
    components_dir.mkdir(exist_ok=True)

    # Create VoiceChat component
    voice_chat_component = """import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, MicOff, Play, Pause, Volume2, Send } from 'lucide-react';

interface VoiceChatProps {
  sector: string;
  onMessage: (message: any) => void;
}

export const VoiceChat: React.FC<VoiceChatProps> = ({ sector, onMessage }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [llmResponse, setLlmResponse] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [textInput, setTextInput] = useState('');
  
  const websocketRef = useRef<WebSocket | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    // Initialize WebSocket connection
    connectWebSocket();
    
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8765');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    websocketRef.current = ws;
  };

  const handleWebSocketMessage = (data: any) => {
    switch (data.type) {
      case 'transcription':
        setTranscription(data.text);
        break;
      case 'llm_response':
        setLlmResponse(data.text);
        if (data.audio_file) {
          playAudioResponse(data.audio_file);
        }
        onMessage({
          type: 'llm_response',
          text: data.text,
          sector,
          timestamp: new Date().toISOString()
        });
        break;
      case 'recording_started':
        console.log('Recording started');
        break;
      case 'recording_stopped':
        console.log('Recording stopped');
        break;
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        sendAudioToServer(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      
      // Send start recording message
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({ type: 'start_recording' }));
      }
      
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Microphone access denied. Please allow microphone access and try again.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      // Send stop recording message
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({ type: 'stop_recording' }));
      }
    }
  };

  const sendAudioToServer = async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      formData.append('sector', sector);
      
      const response = await fetch('/api/process-audio', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const result = await response.json();
        setTranscription(result.transcription);
        setLlmResponse(result.response);
        
        onMessage({
          type: 'user_audio',
          text: result.transcription,
          sector,
          timestamp: new Date().toISOString()
        });
        
        onMessage({
          type: 'llm_response',
          text: result.response,
          sector,
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      console.error('Error processing audio:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const sendTextMessage = async () => {
    if (!textInput.trim()) return;
    
    setIsProcessing(true);
    
    try {
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({
          type: 'text_input',
          text: textInput
        }));
      }
      
      setTextInput('');
      
      onMessage({
        type: 'user_text',
        text: textInput,
        sector,
        timestamp: new Date().toISOString()
      });
      
    } catch (error) {
      console.error('Error sending text message:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const playAudioResponse = (audioFile: string) => {
    if (audioRef.current) {
      audioRef.current.src = audioFile;
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800">
          🎤 Voice Chat - {sector.replace('_', ' ').toUpperCase()}
        </h2>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-gray-300'}`} />
          <span className="text-sm text-gray-600">
            {isRecording ? 'Recording...' : 'Ready'}
          </span>
        </div>
      </div>

      {/* Voice Recording Controls */}
      <div className="flex justify-center mb-6">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={isRecording ? stopRecording : startRecording}
          className={`w-20 h-20 rounded-full flex items-center justify-center text-white text-2xl transition-all duration-200 ${
            isRecording 
              ? 'bg-red-500 hover:bg-red-600 shadow-lg shadow-red-200' 
              : 'bg-primary-500 hover:bg-primary-600 shadow-lg shadow-primary-200'
          }`}
        >
          {isRecording ? <MicOff /> : <Mic />}
        </motion.button>
      </div>

      {/* Text Input Alternative */}
      <div className="mb-6">
        <div className="flex space-x-2">
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Or type your message here..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            onKeyPress={(e) => e.key === 'Enter' && sendTextMessage()}
          />
          <button
            onClick={sendTextMessage}
            disabled={!textInput.trim() || isProcessing}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Processing Indicator */}
      <AnimatePresence>
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex items-center justify-center py-4"
          >
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
            <span className="ml-2 text-gray-600">Processing...</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Transcription Display */}
      {transcription && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-4 bg-blue-50 rounded-lg border-l-4 border-blue-400"
        >
          <h3 className="font-medium text-blue-800 mb-2">🎵 You said:</h3>
          <p className="text-blue-700">{transcription}</p>
        </motion.div>
      )}

      {/* LLM Response Display */}
      {llmResponse && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-4 bg-green-50 rounded-lg border-l-4 border-green-400"
        >
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium text-green-800">🤖 AI Response:</h3>
            <button
              onClick={() => playAudioResponse(`/audio/response_${Date.now()}.wav`)}
              className="text-green-600 hover:text-green-700"
            >
              <Volume2 className="w-5 h-5" />
            </button>
          </div>
          <p className="text-green-700">{llmResponse}</p>
        </motion.div>
      )}

      {/* Hidden Audio Element */}
      <audio
        ref={audioRef}
        onEnded={handleAudioEnded}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        className="hidden"
      />

      {/* Status Indicators */}
      <div className="flex items-center justify-center space-x-4 text-sm text-gray-500">
        <div className="flex items-center space-x-1">
          <div className={`w-2 h-2 rounded-full ${isRecording ? 'bg-red-500' : 'bg-gray-300'}`} />
          <span>Recording</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-green-500' : 'bg-gray-300'}`} />
          <span>Playing</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-yellow-500' : 'bg-gray-300'}`} />
          <span>Processing</span>
        </div>
      </div>
    </div>
  );
};
"""

    with open(components_dir / "VoiceChat.tsx", "w") as f:
        f.write(voice_chat_component)

    # Create SectorSelector component
    sector_selector_component = """import React from 'react';
import { motion } from 'framer-motion';
import { Building2, Car, Heart, GraduationCap, Scale, Shield, Leaf, Zap } from 'lucide-react';

interface SectorSelectorProps {
  selectedSector: string;
  onSectorChange: (sector: string) => void;
}

const sectors = [
  { id: 'finance_banking', name: 'Finance & Banking', icon: Building2, color: 'blue' },
  { id: 'healthcare', name: 'Healthcare', icon: Heart, color: 'red' },
  { id: 'education', name: 'Education', icon: GraduationCap, color: 'green' },
  { id: 'legal', name: 'Legal Services', icon: Scale, color: 'purple' },
  { id: 'transportation', name: 'Transportation', icon: Car, color: 'orange' },
  { id: 'agriculture', name: 'Agriculture', icon: Leaf, color: 'emerald' },
  { id: 'energy', name: 'Energy', icon: Zap, color: 'yellow' },
  { id: 'defense_security', name: 'Defense & Security', icon: Shield, color: 'gray' },
];

const colorClasses = {
  blue: 'bg-blue-100 text-blue-700 border-blue-200 hover:bg-blue-200',
  red: 'bg-red-100 text-red-700 border-red-200 hover:bg-red-200',
  green: 'bg-green-100 text-green-700 border-green-200 hover:bg-green-200',
  purple: 'bg-purple-100 text-purple-700 border-purple-200 hover:bg-purple-200',
  orange: 'bg-orange-100 text-orange-700 border-orange-200 hover:bg-orange-200',
  emerald: 'bg-emerald-100 text-emerald-700 border-emerald-200 hover:bg-emerald-200',
  yellow: 'bg-yellow-100 text-yellow-700 border-yellow-200 hover:bg-yellow-200',
  gray: 'bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200',
};

export const SectorSelector: React.FC<SectorSelectorProps> = ({ selectedSector, onSectorChange }) => {
  return (
    <div className="card">
      <h3 className="text-xl font-bold text-gray-800 mb-4">🏢 Select Business Sector</h3>
      
      <div className="space-y-3">
        {sectors.map((sector) => {
          const IconComponent = sector.icon;
          const isSelected = selectedSector === sector.id;
          const colorClass = colorClasses[sector.color as keyof typeof colorClasses];
          
          return (
            <motion.button
              key={sector.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onSectorChange(sector.id)}
              className={`w-full p-3 rounded-lg border-2 transition-all duration-200 flex items-center space-x-3 ${
                isSelected 
                  ? 'border-primary-500 bg-primary-50 text-primary-700 shadow-md' 
                  : `${colorClass} border-transparent`
              }`}
            >
              <IconComponent className="w-5 h-5" />
              <span className="font-medium">{sector.name}</span>
              {isSelected && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="ml-auto w-3 h-3 bg-primary-500 rounded-full"
                />
              )}
            </motion.button>
          );
        })}
      </div>
      
      <div className="mt-6 p-3 bg-gray-50 rounded-lg">
        <p className="text-sm text-gray-600">
          <strong>Selected:</strong> {sectors.find(s => s.id === selectedSector)?.name || 'General'}
        </p>
        <p className="text-xs text-gray-500 mt-1">
          The AI will provide specialized responses for this sector
        </p>
      </div>
    </div>
  );
};
"""

    with open(components_dir / "SectorSelector.tsx", "w") as f:
        f.write(sector_selector_component)

    # Create ChatHistory component
    chat_history_component = """import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, User, Bot, Clock } from 'lucide-react';

interface ChatMessage {
  type: 'user_audio' | 'user_text' | 'llm_response';
  text: string;
  sector: string;
  timestamp: string;
}

interface ChatHistoryProps {
  messages: ChatMessage[];
}

export const ChatHistory: React.FC<ChatHistoryProps> = ({ messages }) => {
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('tr-TR', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getMessageIcon = (type: string) => {
    switch (type) {
      case 'user_audio':
      case 'user_text':
        return <User className="w-4 h-4" />;
      case 'llm_response':
        return <Bot className="w-4 h-4" />;
      default:
        return <MessageCircle className="w-4 h-4" />;
    }
  };

  const getMessageColor = (type: string) => {
    switch (type) {
      case 'user_audio':
      case 'user_text':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'llm_response':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  if (messages.length === 0) {
    return (
      <div className="card text-center py-12">
        <MessageCircle className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-xl font-medium text-gray-600 mb-2">No messages yet</h3>
        <p className="text-gray-500">
          Start a voice conversation or type a message to begin chatting with the AI
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="text-xl font-bold text-gray-800 mb-4">💬 Chat History</h3>
      
      <div className="space-y-4 max-h-96 overflow-y-auto">
        <AnimatePresence>
          {messages.map((message, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.1 }}
              className={`p-4 rounded-lg border-l-4 ${getMessageColor(message.type)}`}
            >
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 mt-1">
                  {getMessageIcon(message.type)}
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium capitalize">
                      {message.type === 'llm_response' ? 'AI Assistant' : 'You'}
                    </span>
                    <div className="flex items-center space-x-2 text-xs text-gray-600">
                      <Clock className="w-3 h-3" />
                      <span>{formatTimestamp(message.timestamp)}</span>
                    </div>
                  </div>
                  
                  <p className="text-sm leading-relaxed">{message.text}</p>
                  
                  <div className="mt-2">
                    <span className="inline-block px-2 py-1 text-xs bg-white bg-opacity-50 rounded-full">
                      {message.sector.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
      
      <div className="mt-4 text-center text-sm text-gray-500">
        {messages.length} message{messages.length !== 1 ? 's' : ''} in conversation
      </div>
    </div>
  );
};
"""

    with open(components_dir / "ChatHistory.tsx", "w") as f:
        f.write(chat_history_component)

    # Create Header component
    header_component = """import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Globe, Sparkles } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center space-x-3"
          >
            <div className="w-12 h-12 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center">
              <Brain className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-800">
                Turkish AI Agent
              </h1>
              <p className="text-gray-600 text-sm">
                Intelligent voice assistant for Turkish businesses
              </p>
            </div>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center space-x-4"
          >
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Globe className="w-4 h-4" />
              <span>🇹🇷 Türkçe</span>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Sparkles className="w-4 h-4" />
              <span>AI Powered</span>
            </div>
          </motion.div>
        </div>
      </div>
    </header>
  );
};
"""

    with open(components_dir / "Header.tsx", "w") as f:
        f.write(header_component)

    # Create index.html
    index_html = """<!doctype html>
<html lang="tr">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Turkish AI Agent - Voice Chat Interface</title>
    <meta name="description" content="Intelligent voice assistant for Turkish businesses with sector-specific expertise" />
    
    <!-- Inter font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"""

    with open(ui_dir / "index.html", "w") as f:
        f.write(index_html)

    # Create README
    readme = """# Turkish AI Agent - Voice Chat UI

A modern React-based voice chat interface for the Turkish LLM project.

## Features

- 🎤 Real-time voice recording and transcription
- 🤖 AI-powered responses with sector-specific expertise
- 🔊 Text-to-speech synthesis with SSML support
- 🏢 Sector selection for specialized responses
- 💬 Chat history and conversation tracking
- 📱 Responsive design with Tailwind CSS
- ✨ Smooth animations with Framer Motion

## Quick Start

1. Install dependencies:
   ```bash
   cd ui
   npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

3. Open http://localhost:3000 in your browser

## Prerequisites

- Node.js 16+ and npm
- Voice orchestrator service running on port 8765
- Turkish LLM inference service running on port 8000

## Build for Production

```bash
npm run build
npm run preview
```

## Architecture

- **VoiceChat**: Main voice interaction component
- **SectorSelector**: Business sector selection
- **ChatHistory**: Conversation history display
- **Header**: Application header and branding

## WebSocket Events

- `start_recording`: Begin audio recording
- `stop_recording`: Stop audio recording
- `text_input`: Send text message
- `transcription`: Receive speech transcription
- `llm_response`: Receive AI response

## Styling

Built with Tailwind CSS for consistent, responsive design.
"""

    with open(ui_dir / "README.md", "w") as f:
        f.write(readme)

    print("✅ React voice chat UI components created successfully!")
    print("📁 Files created in the 'ui' directory")
    print("🚀 Run 'cd ui && npm install && npm run dev' to start")


if __name__ == "__main__":
    create_react_components()

import React, { useState, useEffect, useRef, useCallback } from 'react';

interface VoiceChatProps {
  sector: string;
  onMessage: (message: any) => void;
}

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  sector: string;
  audioFile?: string;
  confidence?: number;
}

interface WebSocketMessage {
  type: string;
  message?: string;
  text?: string;
  audio_file?: string;
  sector?: string;
  confidence?: number;
  success?: boolean;
}

export const VoiceChat: React.FC<VoiceChatProps> = ({ sector, onMessage }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('Bağlantı bekleniyor...');
  const [textInput, setTextInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [stats, setStats] = useState<any>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    try {
      const ws = new WebSocket('ws://localhost:8765');
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(' WebSocket connected');
        setIsConnected(true);
        setConnectionError(null);
        setCurrentStatus('Bağlantı kuruldu');
        reconnectAttempts.current = 0;

        // Request initial stats
        ws.send(JSON.stringify({ type: 'get_stats' }));
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error(' Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log(' WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setIsRecording(false);
        setIsProcessing(false);
        
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          setCurrentStatus(`Yeniden bağlanılıyor... (${reconnectAttempts.current + 1}/${maxReconnectAttempts})`);
          reconnectAttempts.current++;
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, 2000 * reconnectAttempts.current);
        } else {
          setCurrentStatus('Bağlantı kesildi');
          setConnectionError('WebSocket bağlantısı kurulamadı');
        }
      };

      ws.onerror = (error) => {
        console.error(' WebSocket error:', error);
        setConnectionError('WebSocket hatası oluştu');
      };

    } catch (error) {
      console.error(' Failed to create WebSocket:', error);
      setConnectionError('WebSocket oluşturulamadı');
    }
  }, []);

  // Handle WebSocket messages
  const handleWebSocketMessage = (data: WebSocketMessage) => {
    switch (data.type) {
      case 'recording_started':
        if (data.success) {
          setCurrentStatus(' Kayıt başladı - Konuşmaya başlayın');
        } else {
          setCurrentStatus(' Kayıt başlatılamadı');
          setIsRecording(false);
        }
        break;

      case 'transcription':
        if (data.text) {
          setCurrentStatus(' Metin algılandı');
          addMessage({
            id: Date.now().toString(),
            type: 'user',
            content: data.text,
            timestamp: new Date(),
            sector
          });
        }
        break;

      case 'llm_response':
        if (data.text) {
          setCurrentStatus(' AI yanıtı alındı');
          addMessage({
            id: Date.now().toString(),
            type: 'assistant',
            content: data.text,
            timestamp: new Date(),
            sector: data.sector || sector,
            confidence: data.confidence
          });
        }
        break;

      case 'audio_response':
        setCurrentStatus(' Ses yanıtı hazır');
        setIsProcessing(false);
        setIsRecording(false);
        
        if (data.audio_file) {
          console.log(' Audio file ready:', data.audio_file);
        }
        break;

      case 'response_complete':
        setCurrentStatus(' İşlem tamamlandı');
        setIsProcessing(false);
        setIsRecording(false);
        break;

      case 'status':
        if (data.message) {
          setCurrentStatus(data.message);
        }
        break;

      case 'error':
        console.error(' Server error:', data.message);
        setCurrentStatus(` Hata: ${data.message}`);
        setIsProcessing(false);
        setIsRecording(false);
        break;

      case 'stats':
        setStats(data.data);
        break;

      default:
        console.log(' Unknown message type:', data.type);
    }
  };

  // Add message to chat
  const addMessage = (message: Message) => {
    setMessages(prev => [...prev, message]);
    onMessage(message);
  };

  // Start voice recording
  const startRecording = async () => {
    if (!isConnected || !wsRef.current) {
      setConnectionError('WebSocket bağlantısı yok');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Initialize audio context for level monitoring
      audioContextRef.current = new AudioContext();
      const analyser = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyser);
      
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateAudioLevel = () => {
        if (isRecording) {
          analyser.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          setAudioLevel(average / 255);
          requestAnimationFrame(updateAudioLevel);
        }
      };
      updateAudioLevel();

      // Start MediaRecorder
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const reader = new FileReader();
        reader.onload = () => {
          const base64Audio = (reader.result as string).split(',')[1];
          if (wsRef.current) {
            wsRef.current.send(JSON.stringify({
              type: 'audio_data',
              audio: base64Audio
            }));
          }
        };
        reader.readAsDataURL(audioBlob);
        
        // Cleanup
        stream.getTracks().forEach(track => track.stop());
        if (audioContextRef.current) {
          audioContextRef.current.close();
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setIsProcessing(true);
      
      // Notify server
      wsRef.current.send(JSON.stringify({ type: 'start_recording' }));
      
    } catch (error) {
      console.error(' Failed to start recording:', error);
      setCurrentStatus(' Mikrofon erişimi reddedildi');
    }
  };

  // Stop voice recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setAudioLevel(0);
      
      if (wsRef.current) {
        wsRef.current.send(JSON.stringify({ type: 'stop_recording' }));
      }
    }
  };

  // Send text message
  const sendTextMessage = () => {
    if (!textInput.trim() || !isConnected || !wsRef.current) return;

    const message: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: textInput,
      timestamp: new Date(),
      sector
    };

    addMessage(message);
    setIsProcessing(true);
    
    wsRef.current.send(JSON.stringify({
      type: 'text_input',
      text: textInput,
      sector_hint: sector
    }));
    
    setTextInput('');
  };

  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [connectWebSocket]);

  // Handle Enter key for text input
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage();
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xl font-bold text-gray-800"> Sesli Sohbet</h2>
          <div className={`flex items-center space-x-2 text-sm ${
            isConnected ? 'text-green-600' : 'text-red-600'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-500' : 'bg-red-500'
            }`}></div>
            <span>{isConnected ? 'Bağlı' : 'Bağlantı Yok'}</span>
          </div>
        </div>
        
        <div className="text-sm text-gray-600">
          {currentStatus}
        </div>
        
        {connectionError && (
          <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
            {connectionError}
            <button 
              onClick={connectWebSocket}
              className="ml-2 text-red-800 underline hover:no-underline"
            >
              Yeniden Bağlan
            </button>
          </div>
        )}
      </div>

      {/* Voice Recording Section */}
      <div className="mb-6">
        <div className="flex flex-col items-center space-y-4">
          {/* Audio Level Indicator */}
          {isRecording && (
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all duration-100"
                style={{ width: `${audioLevel * 100}%` }}
              ></div>
            </div>
          )}
          
          {/* Record Button */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!isConnected || isProcessing}
            className={`w-20 h-20 rounded-full flex items-center justify-center text-2xl transition-all duration-200 ${
              isRecording 
                ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse' 
                : isConnected
                  ? 'bg-blue-500 hover:bg-blue-600 text-white hover:scale-105'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            {isRecording ? '' : ''}
          </button>
          
          <div className="text-sm text-gray-600 text-center">
            {isRecording 
              ? 'Konuşmayı durdurmak için tıklayın' 
              : isConnected 
                ? 'Konuşmaya başlamak için tıklayın'
                : 'Bağlantı bekleniyor...'
            }
          </div>
        </div>
      </div>

      {/* Text Input Section */}
      <div className="mb-6">
        <div className="flex space-x-2">
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Veya buraya yazın..."
            disabled={!isConnected || isProcessing}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />
          <button
            onClick={sendTextMessage}
            disabled={!textInput.trim() || !isConnected || isProcessing}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            
          </button>
        </div>
      </div>

      {/* Recent Messages */}
      {messages.length > 0 && (
        <div className="border-t pt-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Son Mesajlar</h3>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {messages.slice(-3).map((message) => (
              <div key={message.id} className={`text-sm p-2 rounded ${
                message.type === 'user' 
                  ? 'bg-blue-50 text-blue-800' 
                  : 'bg-gray-50 text-gray-800'
              }`}>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium">
                    {message.type === 'user' ? ' Siz' : ' AI'}
                  </span>
                  <span className="text-xs opacity-75">
                    {message.timestamp.toLocaleTimeString('tr-TR')}
                  </span>
                </div>
                <div>{message.content}</div>
                {message.confidence && (
                  <div className="text-xs mt-1 opacity-75">
                    Güven: {(message.confidence * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div className="mt-4 pt-4 border-t">
          <div className="grid grid-cols-2 gap-4 text-xs text-gray-600">
            <div>Oturum: {stats.sessions}</div>
            <div>Mesaj: {stats.messages_processed}</div>
            <div>Çalışma Süresi: {Math.round(stats.uptime_seconds / 60)}dk</div>
            <div>Hata: {stats.errors}</div>
          </div>
        </div>
      )}
    </div>
  );
};

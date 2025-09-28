import React, { useState, useEffect } from 'react';
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

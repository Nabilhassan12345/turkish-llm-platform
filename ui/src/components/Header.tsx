import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white shadow-lg">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="text-4xl">🤖</div>
            <div>
              <h1 className="text-3xl font-bold">Türkçe AI Asistan</h1>
              <p className="text-blue-100 text-sm">
                22 İş Sektöründe Uzmanlaşmış Sesli Sohbet Sistemi
              </p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-6 text-sm">
            <div className="text-center">
              <div className="text-2xl">🏢</div>
              <div className="font-medium">22 Sektör</div>
            </div>
            <div className="text-center">
              <div className="text-2xl">🎤</div>
              <div className="font-medium">Sesli Sohbet</div>
            </div>
            <div className="text-center">
              <div className="text-2xl">⚡</div>
              <div className="font-medium">Hızlı Yanıt</div>
            </div>
          </div>
        </div>
        
        <div className="mt-4 pt-4 border-t border-blue-500">
          <div className="flex flex-wrap items-center justify-center space-x-4 text-xs text-blue-200">
            <span>🚀 RTX 4060 GPU Optimizasyonu</span>
            <span>•</span>
            <span>🎯 QLoRA 4-bit Quantization</span>
            <span>•</span>
            <span>🌍 Türkçe Dil Desteği</span>
            <span>•</span>
            <span>🔧 Mixture of Experts (MoE)</span>
            <span>•</span>
            <span>📊 Gerçek Zamanlı Performans</span>
          </div>
        </div>
      </div>
    </header>
  );
};

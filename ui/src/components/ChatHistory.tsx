import React from 'react';

interface Message {
  type: 'user' | 'assistant';
  content: string;
  sector: string;
  timestamp: string;
}

interface ChatHistoryProps {
  messages: Message[];
}

export const ChatHistory: React.FC<ChatHistoryProps> = ({ messages }) => {
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('tr-TR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getSectorDisplayName = (sectorId: string) => {
    const sectorNames: { [key: string]: string } = {
      'general': 'Genel',
      'finance_banking': 'Finans ve Bankacılık',
      'healthcare': 'Sağlık',
      'education': 'Eğitim',
      'media_publishing': 'Medya ve Yayıncılık',
      'legal': 'Hukuk',
      'public_administration': 'Kamu Yönetimi',
      'manufacturing': 'İmalat Endüstrisi',
      'asset_tracking': 'Varlık Takibi',
      'insurance': 'Sigortacılık',
      'tourism_hospitality': 'Turizm ve Otelcilik',
      'ecommerce': 'E-ticaret',
      'energy': 'Enerji',
      'energy_distribution': 'Enerji Dağıtımı',
      'agriculture': 'Tarım',
      'transportation': 'Ulaştırma ve Lojistik',
      'construction': 'İnşaat ve Yapı',
      'real_estate': 'Gayrimenkul',
      'telecommunications': 'Telekomünikasyon',
      'software_technology': 'Yazılım ve Teknoloji',
      'consulting_services': 'Danışmanlık Hizmetleri',
      'research_development': 'Araştırma ve Geliştirme',
      'environmental_services': 'Çevre Hizmetleri'
    };
    
    return sectorNames[sectorId] || sectorId;
  };

  const getSectorIcon = (sectorId: string) => {
    const sectorIcons: { [key: string]: string } = {
      'general': '🌐',
      'finance_banking': '🏦',
      'healthcare': '🏥',
      'education': '🎓',
      'media_publishing': '📰',
      'legal': '⚖️',
      'public_administration': '🏛️',
      'manufacturing': '🏭',
      'asset_tracking': '📦',
      'insurance': '🛡️',
      'tourism_hospitality': '🏨',
      'ecommerce': '🛒',
      'energy': '⚡',
      'energy_distribution': '🔌',
      'agriculture': '🌾',
      'transportation': '🚚',
      'construction': '🏗️',
      'real_estate': '🏠',
      'telecommunications': '📱',
      'software_technology': '💻',
      'consulting_services': '📊',
      'research_development': '🔬',
      'environmental_services': '🌱'
    };
    
    return sectorIcons[sectorId] || '🏢';
  };

  if (messages.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-center text-gray-500">
          <div className="text-6xl mb-4">💬</div>
          <h3 className="text-lg font-medium mb-2">Henüz mesaj yok</h3>
          <p className="text-sm">
            Sesli sohbet başlatın veya metin mesajı gönderin
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-800 mb-2">💬 Sohbet Geçmişi</h2>
        <p className="text-sm text-gray-600">
          Toplam {messages.length} mesaj
        </p>
      </div>

      <div className="space-y-4 max-h-96 overflow-y-auto">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.type === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-xs lg:max-w-md p-3 rounded-lg ${
                message.type === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-sm">
                  {message.type === 'user' ? '👤 Siz' : '🤖 AI Asistan'}
                </span>
                <span className="text-xs opacity-75">
                  {getSectorIcon(message.sector)} {getSectorDisplayName(message.sector)}
                </span>
              </div>
              
              <div className="text-sm mb-2">
                {message.content}
              </div>
              
              <div className={`text-xs ${
                message.type === 'user' ? 'text-blue-100' : 'text-gray-500'
              }`}>
                {formatTimestamp(message.timestamp)}
              </div>
            </div>
          </div>
        ))}
      </div>

      {messages.length > 0 && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              <span className="font-medium">{messages.length}</span> mesaj kaydedildi
            </div>
            <button
              onClick={() => window.location.reload()}
              className="text-sm text-blue-600 hover:text-blue-800 underline"
            >
              Sohbeti Temizle
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

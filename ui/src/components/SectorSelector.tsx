import React from 'react';

interface SectorSelectorProps {
  selectedSector: string;
  onSectorChange: (sector: string) => void;
}

export const SectorSelector: React.FC<SectorSelectorProps> = ({ selectedSector, onSectorChange }) => {
  const sectors = [
    { id: 'general', name: 'Genel', icon: '🌐', description: 'Genel amaçlı AI asistan' },
    { id: 'finance_banking', name: 'Finans ve Bankacılık', icon: '🏦', description: 'Bankacılık ve finansal hizmetler' },
    { id: 'healthcare', name: 'Sağlık', icon: '🏥', description: 'Tıp ve sağlık hizmetleri' },
    { id: 'education', name: 'Eğitim', icon: '🎓', description: 'Eğitim ve öğretim' },
    { id: 'media_publishing', name: 'Medya ve Yayıncılık', icon: '📰', description: 'Gazete, TV, radyo ve dijital medya' },
    { id: 'legal', name: 'Hukuk', icon: '⚖️', description: 'Hukuki danışmanlık ve yasal hizmetler' },
    { id: 'public_administration', name: 'Kamu Yönetimi', icon: '🏛️', description: 'Devlet kurumları ve kamu hizmetleri' },
    { id: 'manufacturing', name: 'İmalat Endüstrisi', icon: '🏭', description: 'Fabrika ve endüstriyel üretim' },
    { id: 'asset_tracking', name: 'Varlık Takibi', icon: '📦', description: 'Varlık yönetimi ve envanter takibi' },
    { id: 'insurance', name: 'Sigortacılık', icon: '🛡️', description: 'Sigorta şirketleri ve risk yönetimi' },
    { id: 'tourism_hospitality', name: 'Turizm ve Otelcilik', icon: '🏨', description: 'Otel, restoran ve turizm hizmetleri' },
    { id: 'ecommerce', name: 'E-ticaret', icon: '🛒', description: 'Online alışveriş ve dijital satış' },
    { id: 'energy', name: 'Enerji', icon: '⚡', description: 'Enerji üretimi ve yönetimi' },
    { id: 'energy_distribution', name: 'Enerji Dağıtımı', icon: '🔌', description: 'Enerji şebekesi ve iletim sistemleri' },
    { id: 'agriculture', name: 'Tarım', icon: '🌾', description: 'Çiftçilik ve tarımsal üretim' },
    { id: 'transportation', name: 'Ulaştırma ve Lojistik', icon: '🚚', description: 'Kara, deniz, hava taşımacılığı' },
    { id: 'construction', name: 'İnşaat ve Yapı', icon: '🏗️', description: 'Bina, altyapı ve inşaat projeleri' },
    { id: 'real_estate', name: 'Gayrimenkul', icon: '🏠', description: 'Emlak ve gayrimenkul yatırımı' },
    { id: 'telecommunications', name: 'Telekomünikasyon', icon: '📱', description: 'İletişim teknolojileri ve internet' },
    { id: 'software_technology', name: 'Yazılım ve Teknoloji', icon: '💻', description: 'Yazılım geliştirme ve teknoloji çözümleri' },
    { id: 'consulting_services', name: 'Danışmanlık Hizmetleri', icon: '📊', description: 'İş danışmanlığı ve strateji' },
    { id: 'research_development', name: 'Araştırma ve Geliştirme', icon: '🔬', description: 'Bilimsel araştırma ve teknoloji geliştirme' },
    { id: 'environmental_services', name: 'Çevre Hizmetleri', icon: '🌱', description: 'Çevre koruma ve sürdürülebilirlik' }
  ];

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-800 mb-2">🏢 Sektör Seçimi</h2>
        <p className="text-sm text-gray-600">
          Uzmanlaşmak istediğiniz iş sektörünü seçin
        </p>
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto">
        {sectors.map((sector) => (
          <button
            key={sector.id}
            onClick={() => onSectorChange(sector.id)}
            className={`w-full text-left p-3 rounded-lg border transition-all duration-200 hover:shadow-md ${
              selectedSector === sector.id
                ? 'border-blue-500 bg-blue-50 text-blue-800'
                : 'border-gray-200 hover:border-gray-300 text-gray-700'
            }`}
          >
            <div className="flex items-center space-x-3">
              <span className="text-2xl">{sector.icon}</span>
              <div className="flex-1">
                <div className="font-medium">{sector.name}</div>
                <div className="text-xs text-gray-500">{sector.description}</div>
              </div>
              {selectedSector === sector.id && (
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              )}
            </div>
          </button>
        ))}
      </div>

      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <h4 className="text-sm font-medium text-blue-800 mb-2">💡 Bilgi:</h4>
        <p className="text-xs text-blue-700">
          Seçilen sektöre göre AI asistanınız o alanda uzmanlaşmış yanıtlar verecektir. 
          Her sektör için özel eğitilmiş adaptörler kullanılmaktadır.
        </p>
      </div>
    </div>
  );
};

import React, { useState } from 'react';
import { X, Clock, Play, Headphones, Heart, Star, ChevronRight, Share2 } from 'lucide-react';

const ExitRecommendationModal = () => {
  const [visible, setVisible] = useState(true);
  const [selectedTab, setSelectedTab] = useState('foryou');
  
  const closeModal = () => {
    setVisible(false);
  };
  
  const mockRecommendations = {
    foryou: [
      {
        title: "The Science of Productivity",
        creator: "Dr. Maya Patel",
        duration: "12 min",
        category: "Self Development",
        matchPercentage: 97,
        image: "/api/placeholder/80/80"
      },
      {
        title: "Midnight Mystery: Episode 3",
        creator: "Kuku Originals",
        duration: "8 min",
        category: "Fiction",
        matchPercentage: 94,
        image: "/api/placeholder/80/80"
      },
      {
        title: "Quick Financial Tips for Beginners",
        creator: "Money Matters",
        duration: "6 min",
        category: "Finance",
        matchPercentage: 91,
        image: "/api/placeholder/80/80"
      }
    ],
    trending: [
      {
        title: "History's Greatest Conspiracies",
        creator: "Unknown Truths",
        duration: "15 min",
        category: "History",
        likes: "2.4k",
        image: "/api/placeholder/80/80"
      },
      {
        title: "Morning Meditation Routine",
        creator: "Calm Mind",
        duration: "5 min",
        category: "Wellness",
        likes: "1.8k",
        image: "/api/placeholder/80/80"
      },
      {
        title: "Tech News Roundup",
        creator: "Digital Pulse",
        duration: "10 min",
        category: "Technology",
        likes: "1.2k",
        image: "/api/placeholder/80/80"
      }
    ],
    continue: [
      {
        title: "Build Better Habits",
        creator: "Life Coach",
        duration: "18 min remaining",
        progress: 42,
        category: "Self Development",
        image: "/api/placeholder/80/80"
      },
      {
        title: "The Art of Negotiation",
        creator: "Success Academy",
        duration: "7 min remaining",
        progress: 75,
        category: "Business",
        image: "/api/placeholder/80/80"
      }
    ]
  };

  if (!visible) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-100">
        <button 
          onClick={() => setVisible(true)}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg shadow-md"
        >
          Show Exit Modal (Swipe Simulation)
        </button>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-xl w-full max-w-md overflow-hidden shadow-xl">
        {/* Header */}
        <div className="relative bg-gradient-to-r from-purple-700 to-indigo-800 px-6 py-4">
          <button 
            onClick={closeModal}
            className="absolute right-4 top-4 text-white opacity-80 hover:opacity-100"
          >
            <X size={20} />
          </button>
          <h2 className="text-xl font-bold text-white">Before You Leave...</h2>
          <p className="text-purple-100 text-sm mt-1">
            We've curated some quick listens just for you
          </p>
        </div>
        
        {/* Tab Navigation */}
        <div className="flex border-b">
          <button 
            className={`flex-1 py-3 text-sm font-medium ${selectedTab === 'foryou' ? 'text-purple-700 border-b-2 border-purple-700' : 'text-gray-500'}`}
            onClick={() => setSelectedTab('foryou')}
          >
            For You
          </button>
          <button 
            className={`flex-1 py-3 text-sm font-medium ${selectedTab === 'trending' ? 'text-purple-700 border-b-2 border-purple-700' : 'text-gray-500'}`}
            onClick={() => setSelectedTab('trending')}
          >
            Trending
          </button>
          <button 
            className={`flex-1 py-3 text-sm font-medium ${selectedTab === 'continue' ? 'text-purple-700 border-b-2 border-purple-700' : 'text-gray-500'}`}
            onClick={() => setSelectedTab('continue')}
          >
            Continue
          </button>
        </div>
        
        {/* Content */}
        <div className="p-4 max-h-96 overflow-y-auto">
          {selectedTab === 'foryou' && (
            <div>
              <p className="text-xs text-gray-500 mb-3">Based on your listening history</p>
              {mockRecommendations.foryou.map((item, index) => (
                <div key={index} className="flex items-center p-2 mb-3 rounded-lg hover:bg-purple-50 cursor-pointer">
                  <div className="relative">
                    <img src={item.image} alt={item.title} className="w-16 h-16 rounded-lg object-cover" />
                    <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-30 rounded-lg opacity-0 hover:opacity-100 transition-opacity">
                      <Play size={24} className="text-white" />
                    </div>
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="flex items-center">
                      <span className="text-xs px-2 py-0.5 bg-purple-100 text-purple-800 rounded-full">{item.matchPercentage}% Match</span>
                    </div>
                    <h3 className="font-medium text-sm mt-1">{item.title}</h3>
                    <p className="text-xs text-gray-500">{item.creator}</p>
                    <div className="flex items-center mt-1 text-xs text-gray-500">
                      <Clock size={12} className="mr-1" /> {item.duration} • {item.category}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {selectedTab === 'trending' && (
            <div>
              <p className="text-xs text-gray-500 mb-3">Popular on Kuku FM right now</p>
              {mockRecommendations.trending.map((item, index) => (
                <div key={index} className="flex items-center p-2 mb-3 rounded-lg hover:bg-purple-50 cursor-pointer">
                  <div className="relative">
                    <img src={item.image} alt={item.title} className="w-16 h-16 rounded-lg object-cover" />
                    <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-30 rounded-lg opacity-0 hover:opacity-100 transition-opacity">
                      <Play size={24} className="text-white" />
                    </div>
                  </div>
                  <div className="ml-3 flex-1">
                    <h3 className="font-medium text-sm">{item.title}</h3>
                    <p className="text-xs text-gray-500">{item.creator}</p>
                    <div className="flex items-center mt-1 text-xs text-gray-500">
                      <Clock size={12} className="mr-1" /> {item.duration} • {item.category}
                    </div>
                    <div className="flex items-center mt-1">
                      <Heart size={12} className="text-red-500 mr-1" /> 
                      <span className="text-xs text-gray-500">{item.likes}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {selectedTab === 'continue' && (
            <div>
              <p className="text-xs text-gray-500 mb-3">Pick up where you left off</p>
              {mockRecommendations.continue.map((item, index) => (
                <div key={index} className="flex items-center p-2 mb-3 rounded-lg hover:bg-purple-50 cursor-pointer">
                  <div className="relative">
                    <img src={item.image} alt={item.title} className="w-16 h-16 rounded-lg object-cover" />
                    <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-30 rounded-lg opacity-0 hover:opacity-100 transition-opacity">
                      <Play size={24} className="text-white" />
                    </div>
                  </div>
                  <div className="ml-3 flex-1">
                    <h3 className="font-medium text-sm">{item.title}</h3>
                    <p className="text-xs text-gray-500">{item.creator}</p>
                    <div className="flex w-full bg-gray-200 rounded-full h-1.5 mt-2">
                      <div 
                        className="bg-purple-600 h-1.5 rounded-full" 
                        style={{ width: `${item.progress}%` }}
                      ></div>
                    </div>
                    <div className="flex items-center mt-1 text-xs text-gray-500">
                      <Clock size={12} className="mr-1" /> {item.duration}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Footer */}
        <div className="p-4 border-t">
          <button className="w-full py-2.5 bg-purple-700 text-white rounded-lg font-medium text-sm hover:bg-purple-800 transition-colors">
            Play Similar Content
          </button>
          <button className="w-full py-2.5 mt-2 border border-gray-300 text-gray-700 rounded-lg font-medium text-sm hover:bg-gray-50 transition-colors" onClick={closeModal}>
            Exit App
          </button>
        </div>
      </div>
    </div>
  );
};

export default ExitRecommendationModal;
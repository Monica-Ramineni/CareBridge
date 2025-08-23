"use client";

import { useState, useEffect, useRef } from "react";
import { apiUrl } from "@/lib/api";



interface Citation {
  text: string;
  url: string;
}

interface Message {
  id: number;
  text: string;
  sender: "user" | "assistant";
  timestamp: string;
  citations?: Citation[];
  triage?: { level: "routine" | "urgent" | "emergency"; reasons: string[] };
  cost?: { estimated_cost?: number; tokens?: number };
  latency?: number;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  timestamp: string;
}

type ConnectionStatus = "online" | "connecting" | "offline" | "error";

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isClient, setIsClient] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("connecting");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);
  const [editingConversationId, setEditingConversationId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [copiedRefsFor, setCopiedRefsFor] = useState<number | null>(null);
  const [debugMode, setDebugMode] = useState(false);

  // Add loading message cycling
  const loadingMessages = [
    "🔍 Analyzing your health question...",
    "📚 Searching medical databases...",
    "💡 Compiling comprehensive response..."
  ];

  const cycleLoadingMessages = () => {
    let messageIndex = 0;
    const interval = setInterval(() => {
      setLoadingMessage(loadingMessages[messageIndex]);
      messageIndex = (messageIndex + 1) % loadingMessages.length;
    }, 1500); // Change message every 1.5 seconds
    
    return interval;
  };

  useEffect(() => {
    setIsClient(true);
    // Initialize with welcome message only on client
    const welcomeMessage: Message = {
      id: 1,
      text: "Hello! I'm your Personal Health Copilot. How can I help you today?",
      sender: "assistant",
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages([welcomeMessage]);
    
    // Set initial conversation ID
    setCurrentConversationId(`conv-${Date.now()}`);
    
    // Start connection monitoring
    checkBackendConnection();
    const interval = setInterval(checkBackendConnection, 900000); // Check every 15 minutes
    
    // Detect debug flag (?debug=1) for developer diagnostics
    try {
      const params = new URLSearchParams(window.location.search);
      if (params.get('debug') === '1') setDebugMode(true);
    } catch (_) {}

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Handle shared chat links
    if (isClient) {
      const urlParams = new URLSearchParams(window.location.search);
      const sharedData = urlParams.get('share');
      if (sharedData) {
        try {
          // Decode the shared data
          const decodedData = JSON.parse(atob(sharedData.replace(/-/g, '+').replace(/_/g, '/')));
          
          // Create a new conversation from the shared data
          const sharedConversation: Conversation = {
            id: decodedData.id,
            title: decodedData.title,
            messages: decodedData.messages,
            timestamp: decodedData.timestamp
          };
          
          // Add to conversations if it doesn't exist
          setConversations(prev => {
            const exists = prev.find(conv => conv.id === sharedConversation.id);
            if (!exists) {
              return [...prev, sharedConversation];
            }
            return prev;
          });
          
          // Load the shared conversation
          loadConversation(sharedConversation);
          
          // Clear the URL parameter
          window.history.replaceState({}, document.title, window.location.pathname);
        } catch (error) {
          console.error('Error loading shared chat:', error);
        }
      }
    }
  }, [isClient]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (activeDropdown && !(event.target as Element).closest('.dropdown-container')) {
        setActiveDropdown(null);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [activeDropdown]);

  // Auto-scroll to bottom when messages change
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const checkBackendConnection = async () => {
    try {
      setConnectionStatus("connecting");
      
      const response = await fetch(apiUrl('/health'), {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      
      if (response.ok) {
        setConnectionStatus("online");
      } else {
        setConnectionStatus("offline");
      }
    } catch (error) {
      console.error('Backend connection error:', error);
      setConnectionStatus("error");
    }
  };

  // Extract markdown links as citations: [text](url)
  const extractCitations = (markdown: string): Citation[] => {
    const regex = /\[([^\]]+)\]\(([^)]+)\)/g;
    const citations: Citation[] = [];
    let match: RegExpExecArray | null;
    while ((match = regex.exec(markdown)) !== null) {
      citations.push({ text: match[1], url: match[2] });
    }
    return citations;
  };

  // Detect greetings/non-health trivial inputs to suppress triage UI
  const isGreetingMessage = (text: string): boolean => {
    if (!text) return false;
    const t = text.toLowerCase().trim();
    const greetings = [
      'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
      'thanks', 'thank you', 'yo', 'hiya', 'hi there', 'hello there'
    ];
    return greetings.some(g => t === g || t.startsWith(g + ' ') || t.endsWith(' ' + g));
  };

  // Heuristic: detect if a message is likely health-related.
  // We keep this conservative to avoid false positives.
  const isLikelyHealthRelated = (text: string): boolean => {
    if (!text) return false;
    const t = text.toLowerCase();
    const keywords = [
      'symptom', 'symptoms', 'pain', 'ache', 'fever', 'cough', 'rash', 'infection',
      'injury', 'wound', 'fracture', 'bleed', 'diagnos', 'treat', 'therapy', 'medicat',
      'drug', 'prescription', 'dose', 'dosage', 'side effect', 'contraindication',
      'allergy', 'asthma', 'diabetes', 'hypertension', 'cholesterol', 'a1c', 'ldl', 'hdl',
      'blood', 'blood pressure', 'bp', 'lab', 'test', 'results', 'report', 'x-ray', 'mri',
      'diet', 'exercise', 'sleep', 'weight', 'vitamin', 'supplement', 'doctor', 'clinic',
      'hospital', 'surgery', 'procedure', 'triage'
    ];
    return keywords.some(k => t.includes(k));
  };

  // Sanitize assistant text: remove inline links/lists and bare URLs
  const sanitizeAssistantText = (text: string): string => {
    if (!text) return "";
    let out = text;
    out = out.replace(/^\s*-\s*\[[^\]]+\]\([^\)]+\)\s*$/gm, "");
    out = out.replace(/^[\t ]*For more detailed information[^\n]*\n?/im, "");
    out = out.replace(/https?:\/\/\S+/g, "");
    out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1");
    out = out.replace(/\n{3,}/g, "\n\n");
    const trimmed = out.trim();
    // If sanitization removed everything, fall back to original text
    return trimmed.length > 0 ? trimmed : text;
  };



  const getStatusDisplay = () => {
    switch (connectionStatus) {
      case "online":
        return { text: "Online", color: "bg-green-500", dotColor: "bg-green-500" };
      case "connecting":
        return { text: "Connecting...", color: "bg-yellow-500", dotColor: "bg-yellow-500" };
      case "offline":
        return { text: "Offline", color: "bg-red-500", dotColor: "bg-red-500" };
      case "error":
        return { text: "Error", color: "bg-red-500", dotColor: "bg-red-500" };
      default:
        return { text: "Unknown", color: "bg-gray-500", dotColor: "bg-gray-500" };
    }
  };

  const startNewConversation = () => {
    // Save current conversation if it has user messages
    if (messages.length > 1) {
      saveCurrentConversation();
    }
    
    const newConversationId = Date.now().toString();
    setCurrentConversationId(newConversationId);
    setMessages([
      {
        id: 1,
        text: "Hello! I'm your Personal Health Copilot. How can I help you today?",
        sender: "assistant",
        timestamp: new Date().toLocaleTimeString()
      }
    ]);
  };

  const generateConversationTitle = async (messages: Message[]): Promise<string> => {
    try {
      // Try LLM-powered title generation first
      const response = await fetch(apiUrl('/generate-title'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: messages.map(msg => ({
            sender: msg.sender,
            text: msg.text
          }))
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        return data.title || "Health Consultation";
      }
    } catch (error) {
      console.log('LLM title generation failed, using fallback method');
    }
    
    // Fallback to existing keyword-based method
    const userMessages = messages.filter(m => m.sender === "user");
    
    if (userMessages.length === 0) return "New Conversation";
    
    // Combine all user messages for analysis
    const allUserText = userMessages.map(m => m.text.toLowerCase()).join(" ");
    
    // Analyze conversation patterns and content
    const analysis = analyzeConversationContent(messages);
    
    // Generate title based on analysis
    return generateSmartTitle(analysis, allUserText);
  };

  const analyzeConversationContent = (messages: Message[]) => {
    const userMessages = messages.filter(m => m.sender === "user");
    
    // Extract key information
    const allText = messages.map(m => m.text.toLowerCase()).join(" ");
    const userText = userMessages.map(m => m.text.toLowerCase()).join(" ");
    
    // Detect conversation patterns
    const patterns = {
      isQuestion: userMessages.some(m => m.text.toLowerCase().includes("what") || m.text.toLowerCase().includes("how") || m.text.toLowerCase().includes("why") || m.text.toLowerCase().includes("when") || m.text.toLowerCase().includes("where")),
      isEmergency: allText.includes("emergency") || allText.includes("urgent") || allText.includes("immediate") || allText.includes("critical"),
      isPrevention: allText.includes("prevent") || allText.includes("avoid") || allText.includes("precaution") || allText.includes("protection"),
      isTreatment: allText.includes("treat") || allText.includes("cure") || allText.includes("therapy") || allText.includes("medication"),
      isDiagnosis: allText.includes("symptom") || allText.includes("diagnosis") || allText.includes("condition") || allText.includes("disease"),
      isLifestyle: allText.includes("diet") || allText.includes("exercise") || allText.includes("sleep") || allText.includes("stress"),
      isMentalHealth: allText.includes("anxiety") || allText.includes("depression") || allText.includes("mood") || allText.includes("stress") || allText.includes("mental"),
      isPetHealth: allText.includes("dog") || allText.includes("cat") || allText.includes("pet") || allText.includes("animal") || allText.includes("veterinary") || allText.includes("vet"),
      isChronic: allText.includes("chronic") || allText.includes("long-term") || allText.includes("ongoing") || allText.includes("persistent"),
      isAcute: allText.includes("sudden") || allText.includes("acute") || allText.includes("immediate") || allText.includes("recent"),
      isFamily: allText.includes("child") || allText.includes("baby") || allText.includes("pregnancy") || allText.includes("elderly") || allText.includes("senior"),
      isSpecificCondition: extractSpecificConditions(allText),
      messageCount: userMessages.length,
      conversationLength: messages.length
    };
    
    return patterns;
  };

  const extractSpecificConditions = (text: string): string[] => {
    const conditions = [
      "diabetes", "hypertension", "asthma", "arthritis", "cancer", "hiv", "aids", "covid", "flu", "cold",
      "headache", "migraine", "back pain", "chest pain", "stomach pain", "heart disease", "lung disease",
      "skin condition", "eye problem", "dental issue", "bone fracture", "infection", "allergy", "obesity",
      "anxiety", "depression", "insomnia", "fatigue", "nausea", "dizziness", "fever", "cough", "sore throat"
    ];
    
    return conditions.filter(condition => text.includes(condition));
  };

  const generateSmartTitle = (analysis: Record<string, unknown>, userText: string): string => {
    const { isQuestion, isEmergency, isPrevention, isTreatment, isDiagnosis, isLifestyle, isMentalHealth, isPetHealth, isChronic, isAcute, isFamily, isSpecificCondition, messageCount } = analysis;
    
    // Priority-based title generation
    if (isEmergency) {
      return "Emergency Health Concern";
    }
    
    // Check if user specifically asked about pet health
    if (isPetHealth) {
      if (userText.includes("dog")) {
        if (isPrevention) return "Dog Health Prevention";
        if (isTreatment) return "Dog Health Treatment";
        return "Dog Health Care";
      }
      if (userText.includes("cat")) {
        if (isPrevention) return "Cat Health Prevention";
        if (isTreatment) return "Cat Health Treatment";
        return "Cat Health Care";
      }
      if (isPrevention) return "Pet Health Prevention";
      if (isTreatment) return "Pet Health Treatment";
      return "Pet Health Care";
    }
    
    // Check if user specifically asked about lifestyle
    if (userText.includes("lifestyle") || userText.includes("healthy") || userText.includes("diet") || userText.includes("exercise")) {
      return "Healthy Lifestyle";
    }
    
    // Check if user specifically asked about mental health
    if (userText.includes("anxiety") || userText.includes("depression") || userText.includes("stress") || userText.includes("mental")) {
      if (isPrevention) return "Mental Health Prevention";
      if (isTreatment) return "Mental Health Treatment";
      return "Mental Health Support";
    }
    
    if (Array.isArray(isSpecificCondition) && isSpecificCondition.length > 0) {
      const condition = isSpecificCondition[0] as string;
      if (isPrevention) {
        return `${capitalizeFirst(condition)} Prevention`;
      }
      if (isTreatment) {
        return `${capitalizeFirst(condition)} Treatment`;
      }
      if (isDiagnosis) {
        return `${capitalizeFirst(condition)} Symptoms`;
      }
      return `${capitalizeFirst(condition)} Health`;
    }
    
    if (isLifestyle) {
      if (isPrevention) return "Lifestyle Health Prevention";
      return "Healthy Lifestyle";
    }
    
    if (isMentalHealth) {
      if (isPrevention) return "Mental Health Prevention";
      if (isTreatment) return "Mental Health Treatment";
      return "Mental Health Support";
    }
    
    if (isFamily) {
      if (userText.includes("child") || userText.includes("baby")) return "Pediatric Health";
      if (userText.includes("pregnancy")) return "Pregnancy Health";
      if (userText.includes("elderly") || userText.includes("senior")) return "Senior Health";
      return "Family Health";
    }
    
    if (isChronic) {
      return "Chronic Health Management";
    }
    
    if (isAcute) {
      return "Acute Health Issue";
    }
    
    if (isPrevention) {
      return "Health Prevention";
    }
    
    if (isTreatment) {
      return "Treatment Consultation";
    }
    
    if (isDiagnosis) {
      return "Health Symptoms";
    }
    
    if (isQuestion) {
      // Extract the main topic from the question
      const questionWords = userText.split(/\s+/).filter(word => word.length > 3);
      const keyWords = questionWords.slice(0, 2).join(" ");
      if (keyWords) {
        return `${capitalizeFirst(keyWords)} Question`;
      }
      return "Health Question";
    }
    
    // Fallback: Create a meaningful title from user messages
    if (messageCount === 1) {
      const firstMessage = userText.split(/\s+/).slice(0, 3).join(" ");
      return capitalizeFirst(firstMessage);
    } else {
      // Multiple messages - create a summary
      const uniqueWords = new Set(userText.split(/\s+/).filter(word => word.length > 3));
      const keyWords = Array.from(uniqueWords).slice(0, 2).join(" ");
      return keyWords ? capitalizeFirst(keyWords) : "Health Consultation";
    }
  };

  const capitalizeFirst = (str: string): string => {
    return str.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  const saveCurrentConversation = async () => {
    if (messages.length > 1 && currentConversationId) {
      const title = await generateConversationTitle(messages);
      const conversation: Conversation = {
        id: currentConversationId,
        title: title,
        messages: messages,
        timestamp: new Date().toLocaleString()
      };
      
      setConversations(prev => {
        const existingIndex = prev.findIndex(c => c.id === currentConversationId);
        if (existingIndex >= 0) {
          // Update existing conversation
          const updated = [...prev];
          updated[existingIndex] = conversation;
          return updated;
        } else {
          // Add new conversation
          return [...prev, conversation];
        }
      });
    }
  };

  const loadConversation = (conversation: Conversation) => {
    setCurrentConversationId(conversation.id);
    setMessages(conversation.messages);
  };

  const clearHistory = () => {
    setConversations([]);
    setCurrentConversationId(null);
    setMessages([]);
  };

  const handleRenameConversation = (conversationId: string, newTitle: string) => {
    setConversations(prev => 
      prev.map(conv => 
        conv.id === conversationId 
          ? { ...conv, title: newTitle }
          : conv
      )
    );
    setEditingConversationId(null);
    setEditingTitle('');
  };

  const handleShareConversation = (conversation: Conversation) => {
    // Create a shareable link with compressed chat data
    const shareData = {
      id: conversation.id,
      title: conversation.title,
      messages: conversation.messages,
      timestamp: conversation.timestamp
    };
    // Use a shorter encoding method
    const compressedData = btoa(JSON.stringify(shareData)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
    const shareUrl = `${window.location.origin}?share=${compressedData}`;
    navigator.clipboard.writeText(shareUrl);
    alert('Chat link copied to clipboard!');
  };

  const handleDeleteConversation = (conversationId: string) => {
    setConversations(prev => prev.filter(conv => conv.id !== conversationId));
    if (currentConversationId === conversationId) {
      startNewConversation();
    }
  };

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    
    setIsUploading(true);
    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }
      
      const response = await fetch(apiUrl('/upload-lab-results'), {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
        setUploadedFiles(Array.from(files));
        
        // Add a system message about uploaded files
        const uploadMessage: Message = {
          id: Date.now(),
          text: `✅ Successfully uploaded ${data.files_processed} lab result file(s). You can now ask questions about your results.`,
          sender: 'assistant',
          timestamp: new Date().toLocaleTimeString()
        };
        setMessages(prev => [...prev, uploadMessage]);
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      const errorMessage: Message = {
        id: Date.now(),
        text: "❌ Failed to upload lab results. Please try again.",
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const newMessage: Message = {
      id: messages.length + 1,
      text: inputMessage,
      sender: "user",
      timestamp: new Date().toLocaleTimeString()
    };

    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    setInputMessage("");
    setIsLoading(true);

    let loadingInterval: NodeJS.Timeout | null = null;

    try {
      // Show loading message for all non-greeting questions
      const currentMessage = inputMessage.toLowerCase();
      const greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"];
      const isGreeting = greetings.some(greeting => currentMessage.includes(greeting));
      
      if (!isGreeting) {
        // Start cycling loading messages
        loadingInterval = cycleLoadingMessages();
      }
      
      // Use POST /chat endpoint
      const response = await fetch(apiUrl('/chat'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          user_id: 'default',
          conversation_history: messages.map(msg => ({
            sender: msg.sender,
            text: msg.text
          })),
          session_id: sessionId
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        const citations = extractCitations(data.response || "");
        const assistantResponse: Message = {
          id: Date.now() + 1,
          text: data.response,
          sender: 'assistant',
          timestamp: new Date().toLocaleTimeString(),
          citations,
          triage: data.triage || undefined,
          cost: data.cost || undefined,
          latency: typeof data.latency === 'number' ? data.latency : undefined,
        };
        const finalMessages = [...updatedMessages, assistantResponse];
        setMessages(finalMessages);
      } else {
        throw new Error('Failed to get response');
      }
    } catch (error) {
      console.error('Error:', error);
      const errorResponse: Message = {
        id: Date.now() + 1,
        text: "I'm sorry, I'm having trouble connecting to my medical database right now. Please try again in a moment.",
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString()
      };
      const finalMessages = [...updatedMessages, errorResponse];
      setMessages(finalMessages);
    } finally {
      // Clear loading interval if it exists
      if (loadingInterval) {
        clearInterval(loadingInterval);
      }
      setIsLoading(false);
      setLoadingMessage('');
    }
  };

  if (!isClient) {
    return (
      <div className="flex h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="flex-1 flex items-center justify-center">
          <div className="text-gray-400">Loading...</div>
        </div>
      </div>
    );
  }

  const statusDisplay = getStatusDisplay();

  return (
    <div className="flex h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Sidebar */}
      <aside className={`fixed z-30 inset-y-0 left-0 w-72 bg-white shadow-lg transform transition-transform duration-300 ease-in-out
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 lg:static lg:inset-0`}>
        <div className="flex flex-col h-full">
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Chat History</h2>
            <button
              className="lg:hidden p-2 rounded hover:bg-gray-100"
              onClick={() => setIsSidebarOpen(false)}
              aria-label="Close sidebar"
            >
              <svg className="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <div className="p-4 border-b border-gray-200 space-y-2">
            <button
              onClick={() => {/* TODO: Add search functionality */}}
              className="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium flex items-center justify-center"
            >
              🔍 Search Chats
            </button>
            <button
              onClick={startNewConversation}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
            >
              New Chat
            </button>
          </div>



          <div className="flex-1 overflow-y-auto">
            {conversations.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <span className="mb-2">No conversations yet</span>
                <button 
                  onClick={startNewConversation}
                  className="text-sm text-blue-600 hover:bg-blue-50 px-4 py-2 rounded transition"
                >
                  Start New Chat
                </button>
              </div>
            ) : (
              <div className="p-4 space-y-2">
                {conversations.map((conversation) => (
                  <div
                    key={conversation.id}
                    className={`w-full p-3 rounded-lg transition-colors ${
                      currentConversationId === conversation.id
                        ? 'bg-blue-50 border border-blue-200'
                        : 'hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <button
                        onClick={() => loadConversation(conversation)}
                        className="flex-1 text-left min-w-0"
                      >
                        {editingConversationId === conversation.id ? (
                          <input
                            type="text"
                            value={editingTitle}
                            onChange={(e) => setEditingTitle(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') {
                                handleRenameConversation(conversation.id, editingTitle);
                              } else if (e.key === 'Escape') {
                                setEditingConversationId(null);
                                setEditingTitle('');
                              }
                            }}
                            onBlur={() => handleRenameConversation(conversation.id, editingTitle)}
                            className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-blue-500 text-gray-900 bg-white"
                            autoFocus
                          />
                        ) : (
                          <>
                            <div className="font-medium text-gray-900 truncate max-w-full overflow-hidden" title={conversation.title}>{conversation.title}</div>
                            <div className="text-xs text-gray-500 mt-1">{conversation.timestamp}</div>
                          </>
                        )}
                      </button>
                      <div className="relative ml-2 dropdown-container">
                        <button
                          onClick={() => setActiveDropdown(activeDropdown === conversation.id ? null : conversation.id)}
                          className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
                          aria-label="More options"
                        >
                          ⋮
                        </button>
                        {activeDropdown === conversation.id && (
                          <div className="absolute right-0 top-full mt-1 w-32 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
                            <button
                              onClick={() => {
                                setEditingConversationId(conversation.id);
                                setEditingTitle(conversation.title);
                                setActiveDropdown(null);
                              }}
                              className="w-full px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 border-b border-gray-100"
                            >
                              ✏️ Rename
                            </button>
                            <button
                              onClick={() => handleShareConversation(conversation)}
                              className="w-full px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 border-b border-gray-100"
                            >
                              📤 Share
                            </button>
                            <button
                              onClick={() => handleDeleteConversation(conversation.id)}
                              className="w-full px-3 py-2 text-left text-sm text-red-600 hover:bg-red-50"
                            >
                              🗑️ Delete
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          
          {/* File Upload Section - Moved to bottom */}
          <div className="p-4 border-t border-gray-200">
            <h3 className="text-sm font-semibold mb-2 text-gray-900">Upload Lab Results</h3>
            <div className="text-xs text-gray-600 mb-2">
              Supported: PDF, DOCX (Text-based files only)
            </div>
            <input
              type="file"
              multiple
              accept=".pdf,.docx"
              onChange={(e) => handleFileUpload(e.target.files)}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className={`block w-full text-center py-2 px-3 rounded cursor-pointer transition-colors text-sm ${
                isUploading 
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              {isUploading ? 'Uploading...' : 'Upload Lab Results'}
            </label>
            {uploadedFiles.length > 0 && (
              <div className="mt-2 flex items-center justify-between">
                <div className="text-xs text-green-600">
                  ✅ {uploadedFiles.length} file(s) uploaded
                </div>
                <button
                  onClick={() => { setUploadedFiles([]); setSessionId(null); }}
                  className="text-xs text-red-600 hover:bg-red-50 px-2 py-1 rounded"
                  type="button"
                >
                  Clear
                </button>
              </div>
            )}
          </div>
          
          {conversations.length > 0 && (
            <div className="p-4 border-t border-gray-200">
              <button 
                onClick={clearHistory}
                className="w-full text-sm text-red-600 hover:bg-red-50 px-4 py-2 rounded transition"
              >
                Clear History
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* Overlay for mobile */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-40 z-20 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between bg-white shadow-sm border-b border-gray-200 px-8 py-6">
          <div className="flex items-center space-x-4">
            <button
              className="lg:hidden p-2 rounded hover:bg-gray-100"
              onClick={() => setIsSidebarOpen(true)}
              aria-label="Open sidebar"
            >
              <svg className="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <div className="flex items-center">
              <img
                src="/carebridge-logo-white.png?v=1"
                alt="CareBridge"
                width={80}
                height={80}
                className="mr-3 h-20 w-20 object-contain select-none"
                draggable={false}
              />
              <div>
                <h1 className="text-2xl font-bold text-blue-700 tracking-tight">CareBridge</h1>
                <p className="text-sm text-gray-600">Your bridge from worry to well‑informed care</p>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {/* Privacy badge */}
            <span className="hidden sm:inline px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-700 mr-2 border border-gray-200">
              No PHI stored
            </span>
            <span className={`w-3 h-3 ${statusDisplay.dotColor} rounded-full ${connectionStatus === "connecting" ? "animate-pulse" : ""}`} />
            <span className="text-sm text-gray-600">{statusDisplay.text}</span>
          </div>
        </header>

        {/* Chat Area */}
        <main className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0">
            {messages.map((message, index) => {
              const previousUser = [...messages].slice(0, index).reverse().find(m => m.sender === 'user');
              const previousUserText = previousUser ? previousUser.text : '';
              const showTriage = message.sender === 'assistant' && !!message.triage && !isGreetingMessage(previousUserText) && isLikelyHealthRelated(previousUserText);
              const triage = message.triage;
              const triageLevel = triage?.level;
              const triageReasons = triage?.reasons || [];
              const triageClass = triageLevel === 'emergency'
                ? 'bg-red-50 border-red-200 text-red-700'
                : triageLevel === 'urgent'
                  ? 'bg-yellow-50 border-yellow-200 text-yellow-700'
                  : 'bg-green-50 border-green-200 text-green-700';
              return (
              <div key={message.id}>
                <div
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
                      message.sender === 'user'
                        ? 'bg-blue-100 text-gray-900 font-semibold border border-blue-300'
                        : 'bg-white text-gray-900 shadow-sm border border-gray-200'
                    }`}
                  >
                    {message.sender === 'assistant' ? (
                      <div className="text-sm whitespace-pre-wrap">
                        {sanitizeAssistantText(message.text)}
                      </div>
                    ) : (
                      <div 
                        className="text-sm whitespace-pre-wrap"
                        dangerouslySetInnerHTML={{ 
                          __html: message.text
                            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, 
                              '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 underline">$1</a>'
                            )
                        }}
                      />
                    )}
                    <p className={`text-xs mt-1 ${
                      message.sender === 'user' ? 'text-gray-600' : 'text-gray-500'
                    }`}>
                      {message.timestamp}
                    </p>
                  </div>
                </div>

                {/* Activity timeline and citations for assistant replies */}
                {message.sender === 'assistant' && (
                  <div className="ml-8 mt-2 space-y-2">
                    {showTriage && (
                      <div className={`px-3 py-2 rounded text-xs font-medium max-w-xs lg:max-w-md border shadow-sm ${triageClass}`}>
                        {triageLevel === 'emergency' && '🚨 Emergency: '} 
                        {triageLevel === 'urgent' && '⚠️ Urgent: '} 
                        {triageLevel === 'routine' && '✅ Routine: '}
                        {(triageReasons[0]) || 'General guidance.'}
                      </div>
                    )}
                    {showTriage && triageReasons.length > 0 && (
                      <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-sm max-w-xs lg:max-w-md">
                        <div className="text-xs font-semibold text-gray-800 mb-2">Red flags checklist</div>
                        <ul className="space-y-1 text-xs text-gray-700">
                          {triageReasons.map((r, i) => (
                            <li key={`rf-${message.id}-${i}`} className="flex items-start gap-2">
                              <input type="checkbox" className="mt-0.5" />
                              <span>{r}</span>
                            </li>
                          ))}
                        </ul>
                        <div className="text-[10px] text-gray-500 mt-2">Advisory only; not a diagnosis.</div>
                      </div>
                    )}

                    {message.citations && message.citations.length > 0 && (
                      <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-sm max-w-xs lg:max-w-md">
                        <div className="text-xs font-semibold text-gray-800 mb-2">References</div>
                        <ul className="list-disc pl-4 space-y-1">
                          {message.citations.map((c, i) => (
                            <li key={`${c.url}-${i}`} className="text-xs">
                              <a href={c.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 underline">{c.text}</a>
                            </li>
                          ))}
                        </ul>
                        <div className="mt-2 flex gap-2">
                          <button
                            onClick={async () => {
                              const text = (message.citations || []).map(c => `- ${c.text} (${c.url})`).join('\n');
                              try {
                                await navigator.clipboard.writeText(text);
                                setCopiedRefsFor(message.id);
                                setTimeout(() => setCopiedRefsFor(null), 1500);
                              } catch (_) {}
                            }}
                            className="px-2 py-1 text-[11px] bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
                          >
                            {copiedRefsFor === message.id ? 'Copied!' : 'Copy'}
                          </button>
                          <button
                            onClick={() => {
                              const text = (message.citations || []).map(c => `- ${c.text} (${c.url})`).join('\n');
                              const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' });
                              const url = URL.createObjectURL(blob);
                              const a = document.createElement('a');
                              a.href = url;
                              a.download = `carebridge-references-${message.id}.md`;
                              document.body.appendChild(a);
                              a.click();
                              a.remove();
                              URL.revokeObjectURL(url);
                            }}
                            className="px-2 py-1 text-[11px] bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
                          >
                            Download .md
                          </button>
                        </div>
                      </div>
                    )}
                    {debugMode && (message.latency || message.cost) && (
                      <details className="bg-white border border-gray-200 rounded-lg p-3 shadow-sm max-w-xs lg:max-w-md text-xs text-gray-600">
                        <summary className="cursor-pointer select-none">Diagnostics</summary>
                        <div className="mt-2">
                          {typeof message.latency === 'number' && <div>Time: {message.latency.toFixed(2)}s</div>}
                          {message.cost?.estimated_cost !== undefined && <div>Est. cost: ${message.cost.estimated_cost}</div>}
                          {message.cost?.tokens !== undefined && <div>Tokens: {message.cost.tokens}</div>}
                        </div>
                      </details>
                    )}
                  </div>
                )}
              </div>
              );
            })}
            {isLoading && (
              <div className="flex justify-start">
                <div className="max-w-xs lg:max-w-md px-4 py-3 rounded-lg bg-white text-gray-900 shadow-sm border border-gray-200">
                  <p className="text-sm whitespace-pre-wrap">{loadingMessage}</p>
                  <p className="text-xs text-gray-500 mt-1">Loading...</p>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 bg-white p-6 flex-shrink-0">
            {/* Consent note */}
            <div className="text-[11px] text-gray-500 mb-2">
              For safety, don’t share names, phone numbers, or IDs. This is not medical advice.
            </div>
            <form onSubmit={handleSendMessage} className="flex space-x-4">
              <div className="flex-1 relative">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Type your health question here..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-500 text-gray-900"
                  rows={1}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage(e);
                    }
                  }}
                />
              </div>
              <button
                type="submit"
                disabled={!inputMessage.trim() || isLoading}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex-shrink-0"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
}

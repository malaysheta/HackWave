import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { Button } from "@/components/ui/button";

interface Message {
  type: "human" | "ai";
  content: string;
  id: string;
  metadata?: {
    processing_time?: number;
    query_type?: string;
    debate_category?: string;
    domain_analysis?: string;
    ux_analysis?: string;
    technical_analysis?: string;
    revenue_analysis?: string;
    moderator_aggregation?: string;
    debate_resolution?: string;
  };
}

interface StreamEvent {
  type: string;
  content: string;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState<string>("");
  const [streamingMetadata, setStreamingMetadata] = useState<{
    domain_analysis?: string;
    ux_analysis?: string;
    technical_analysis?: string;
    revenue_analysis?: string;
    moderator_aggregation?: string;
    final_answer?: string;
  }>({});
  
  // Add thread management for context persistence
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [conversationHistory, setConversationHistory] = useState<any[]>([]);

  const apiUrl = import.meta.env.DEV
    ? "http://localhost:2024"
    : "http://localhost:2024";

  // Generate or retrieve thread ID
  const getOrCreateThreadId = useCallback(() => {
    if (!currentThreadId) {
      const newThreadId = `frontend_thread_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      setCurrentThreadId(newThreadId);
      return newThreadId;
    }
    return currentThreadId;
  }, [currentThreadId]);

  // Function to get thread ID from URL or create new one
  const getThreadIdFromUrl = useCallback(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const threadId = urlParams.get('thread_id');
    if (threadId) {
      setCurrentThreadId(threadId);
      return threadId;
    }
    return null;
  }, []);

  // Debug: Log current state
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      console.log(`Current state - Thread ID: ${currentThreadId}, Messages: ${messages.length}, Loading: ${isLoading}`);
    }
  }, [currentThreadId, messages.length, isLoading]);

  // Load initial conversation history on app start, but only if no messages exist and no thread ID in URL
  useEffect(() => {
    if (messages.length === 0 && !currentThreadId) {
      const loadInitialHistory = async () => {
        try {
          // Try to load from a default thread or get recent conversations
          const response = await fetch(`${apiUrl}/api/conversation-history/default`, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
            },
          });

          if (response.ok) {
            const history = await response.json();
            if (history && history.length > 0) {
              setConversationHistory(history);
              
              // Convert history to messages format and display immediately
              const historyMessages: Message[] = history.map((entry: any, index: number) => ({
                type: "ai",
                content: entry.final_answer || entry.user_query || "Previous conversation",
                id: entry._id || `history_${Date.now()}_${index}_${Math.random().toString(36).substr(2, 9)}`,
                metadata: {
                  processing_time: entry.processing_time,
                  query_type: entry.query_type,
                  domain_analysis: entry.state_snapshot?.domain_expert_analysis,
                  ux_analysis: entry.state_snapshot?.ux_ui_specialist_analysis,
                  technical_analysis: entry.state_snapshot?.technical_architect_analysis,
                  revenue_analysis: entry.state_snapshot?.revenue_model_analyst_analysis,
                  moderator_aggregation: entry.state_snapshot?.moderator_aggregation,
                },
              }));
              
              setMessages(historyMessages);
              console.log("Loaded conversation history:", historyMessages.length, "messages");
            }
          }
        } catch (error) {
          console.warn("Could not load initial conversation history:", error);
        }
      };

      loadInitialHistory();
    }
  }, [apiUrl, messages.length, currentThreadId]);

  // Function to check if thread has context and load it
  const checkAndLoadThreadContext = useCallback(async (threadId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/thread-context/${threadId}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const contextData = await response.json();
        
        if (contextData.has_context && contextData.history && contextData.history.length > 0) {
          console.log(`Found existing context for thread ${threadId} with ${contextData.history.length} conversations`);
          
          // Convert history to messages format
          const historyMessages: Message[] = contextData.history.map((entry: any, index: number) => ({
            type: "ai",
            content: entry.final_answer || entry.user_query || "Previous conversation",
            id: entry._id || `history_${Date.now()}_${index}_${Math.random().toString(36).substr(2, 9)}`,
            metadata: {
              processing_time: entry.processing_time,
              query_type: entry.query_type,
              domain_analysis: entry.state_snapshot?.domain_expert_analysis,
              ux_analysis: entry.state_snapshot?.ux_ui_specialist_analysis,
              technical_analysis: entry.state_snapshot?.technical_architect_analysis,
              revenue_analysis: entry.state_snapshot?.revenue_model_analyst_analysis,
              moderator_aggregation: entry.state_snapshot?.moderator_aggregation,
            },
          }));
          
          setMessages(historyMessages);
          setConversationHistory(contextData.history);
          return true;
        }
      }
      return false;
    } catch (error) {
      console.warn("Could not check thread context:", error);
      return false;
    }
  }, [apiUrl]);

  // Load initial conversation history on app start, but only if no messages exist and no thread ID in URL
  useEffect(() => {
    if (messages.length === 0 && !currentThreadId) {
      const loadInitialHistory = async () => {
        try {
          // Try to load from a default thread or get recent conversations
          const response = await fetch(`${apiUrl}/api/conversation-history/default`, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
            },
          });

          if (response.ok) {
            const history = await response.json();
            if (history && history.length > 0) {
              setConversationHistory(history);
              
              // Convert history to messages format and display immediately
              const historyMessages: Message[] = history.map((entry: any, index: number) => ({
                type: "ai",
                content: entry.final_answer || entry.user_query || "Previous conversation",
                id: entry._id || `history_${Date.now()}_${index}_${Math.random().toString(36).substr(2, 9)}`,
                metadata: {
                  processing_time: entry.processing_time,
                  query_type: entry.query_type,
                  domain_analysis: entry.state_snapshot?.domain_expert_analysis,
                  ux_analysis: entry.state_snapshot?.ux_ui_specialist_analysis,
                  technical_analysis: entry.state_snapshot?.technical_architect_analysis,
                  revenue_analysis: entry.state_snapshot?.revenue_model_analyst_analysis,
                  moderator_aggregation: entry.state_snapshot?.moderator_aggregation,
                },
              }));
              
              setMessages(historyMessages);
              console.log("Loaded conversation history:", historyMessages.length, "messages");
            }
          }
        } catch (error) {
          console.warn("Could not load initial conversation history:", error);
        }
      };

      loadInitialHistory();
    }
  }, [apiUrl, messages.length, currentThreadId]);

  const loadConversationHistory = async (threadId: string, forceLoad: boolean = false) => {
    try {
      const response = await fetch(`${apiUrl}/api/thread-context/${threadId}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const contextData = await response.json();
        setConversationHistory(contextData.history || []);
        
        // Only set messages if forceLoad is true and we have history
        // This prevents overriding current conversation
        if (forceLoad && contextData.history && contextData.history.length > 0) {
          // Convert history to messages format
          const historyMessages: Message[] = contextData.history.map((entry: any, index: number) => ({
            type: "ai",
            content: entry.final_answer || entry.user_query || "Previous conversation",
            id: entry._id || `history_${Date.now()}_${index}_${Math.random().toString(36).substr(2, 9)}`,
            metadata: {
              processing_time: entry.processing_time,
              query_type: entry.query_type,
              domain_analysis: entry.state_snapshot?.domain_expert_analysis,
              ux_analysis: entry.state_snapshot?.ux_ui_specialist_analysis,
              technical_analysis: entry.state_snapshot?.technical_architect_analysis,
              revenue_analysis: entry.state_snapshot?.revenue_model_analyst_analysis,
              moderator_aggregation: entry.state_snapshot?.moderator_aggregation,
            },
          }));
          
          setMessages(historyMessages);
          console.log(`Loaded ${historyMessages.length} messages from thread context`);
        }
      }
    } catch (error) {
      console.warn("Could not load conversation history:", error);
    }
  };

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [messages, currentStreamingMessage]);

  const handleSubmit = useCallback(
    async (submittedInputValue: string, effort: string) => {
      if (!submittedInputValue.trim()) return;
      
      const threadId = getOrCreateThreadId();
      
      // Check if this thread has existing context and load it if needed
      if (messages.length === 0) {
        const hasContext = await checkAndLoadThreadContext(threadId);
        if (hasContext) {
          console.log("Loaded existing thread context");
        }
      }
      
      setProcessedEventsTimeline([]);
      setIsLoading(true);
      setError(null);
      setCurrentStreamingMessage("");
      setStreamingMetadata({});

      // Add user message with unique ID
      const userMessage: Message = {
        type: "human",
        content: submittedInputValue,
        id: `human_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      };
      
      setMessages(prev => [...prev, userMessage]);

      // Create timeline events for the multi-agent workflow
      const timelineEvents: ProcessedEvent[] = [
        {
          title: "Query Classification",
          data: "Analyzing query type and routing to appropriate specialists...",
        },
        {
          title: "Domain Expert Analysis",
          data: "Analyzing business logic and domain-specific requirements...",
        },
        {
          title: "UX/UI Specialist Analysis", 
          data: "Analyzing user experience and interface design requirements...",
        },
        {
          title: "Technical Architect Analysis",
          data: "Analyzing technical architecture and implementation requirements...",
        },
        {
          title: "Revenue Model Analyst Analysis",
          data: "Analyzing revenue models and monetization strategies...",
        },
        {
          title: "Moderator Aggregation",
          data: "Consolidating feedback and resolving conflicts...",
        },
        {
          title: "Final Answer Generation",
          data: "Generating comprehensive final answer...",
        }
      ];

      // Simulate real-time updates
      for (let i = 0; i < timelineEvents.length; i++) {
        setTimeout(() => {
          setProcessedEventsTimeline(prev => [...prev, timelineEvents[i]]);
        }, i * 1000); // 1 second delay between each event
      }

      try {
        // Use the streaming endpoint with thread_id for context
        const response = await fetch(`${apiUrl}/api/refine-requirements/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: submittedInputValue,
            query_type: effort, // Use effort as query type hint
            thread_id: threadId, // Include thread_id for context persistence
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("No response body reader available");
        }

        const decoder = new TextDecoder();
        let buffer = "";

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const eventData = JSON.parse(line.slice(6));
                  handleStreamEvent(eventData);
                } catch (e) {
                  console.warn('Failed to parse SSE event:', line);
                }
              }
            }
          }
        } finally {
          reader.releaseLock();
        }

        // After completion, don't reload conversation history as it might override current messages
        // The messages are already in the state from the streaming response

      } catch (err) {
        console.error('Streaming error:', err);
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setIsLoading(false);
      }
    },
    [apiUrl, getOrCreateThreadId, loadConversationHistory, checkAndLoadThreadContext, messages.length]
  );

  const handleStreamEvent = useCallback((event: StreamEvent) => {
    switch (event.type) {
      case 'domain_expert':
        setStreamingMetadata((prev: any) => ({ ...prev, domain_analysis: event.content }));
        setCurrentStreamingMessage((prev: string) => prev + "\n\n**Domain Expert Analysis:**\n" + event.content);
        break;
      
      case 'ux_ui_specialist':
        setStreamingMetadata((prev: any) => ({ ...prev, ux_analysis: event.content }));
        setCurrentStreamingMessage((prev: string) => prev + "\n\n**UX/UI Specialist Analysis:**\n" + event.content);
        break;
      
      case 'technical_architect':
        setStreamingMetadata((prev: any) => ({ ...prev, technical_analysis: event.content }));
        setCurrentStreamingMessage((prev: string) => prev + "\n\n**Technical Architect Analysis:**\n" + event.content);
        break;
      
      case 'revenue_model_analyst':
        setStreamingMetadata((prev: any) => ({ ...prev, revenue_analysis: event.content }));
        setCurrentStreamingMessage((prev: string) => prev + "\n\n**Revenue Model Analyst Analysis:**\n" + event.content);
        break;
      
      case 'moderator_aggregation':
        setStreamingMetadata((prev: any) => ({ ...prev, moderator_aggregation: event.content }));
        setCurrentStreamingMessage((prev: string) => prev + "\n\n**Moderator Aggregation:**\n" + event.content);
        break;
      
      case 'final_answer':
        setStreamingMetadata((prev: any) => ({ ...prev, final_answer: event.content }));
        setCurrentStreamingMessage((prev: string) => prev + "\n\n**Final Answer:**\n" + event.content);
        break;
      
      case 'message':
        setCurrentStreamingMessage(event.content);
        break;
      
      case 'complete':
        // Add the final AI message with unique ID
        const aiMessage: Message = {
          type: "ai",
          content: currentStreamingMessage || "Analysis completed",
          id: `ai_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          metadata: {
            ...streamingMetadata,
            processing_time: 0, // Will be calculated properly
            query_type: "general",
          },
        };

        setMessages(prev => [...prev, aiMessage]);

        // Store historical activities
        setHistoricalActivities(prev => ({
          ...prev,
          [aiMessage.id]: [...processedEventsTimeline],
        }));

        // Update URL with thread_id for context persistence
        if (currentThreadId) {
          const url = new URL(window.location.href);
          url.searchParams.set('thread_id', currentThreadId);
          window.history.replaceState({}, '', url.toString());
        }

        // Keep the streaming message visible - it will be cleared when a new analysis starts
        // This ensures the output persists until user clicks "New Analysis"
        break;
      
      case 'error':
        setError(event.content);
        break;
    }
  }, [currentStreamingMessage, streamingMetadata, processedEventsTimeline]);

  const handleCancel = useCallback(() => {
    setIsLoading(false);
    setProcessedEventsTimeline([]);
    setError(null);
    setCurrentStreamingMessage("");
    setStreamingMetadata({});
  }, []);

  const handleNewAnalysis = useCallback(() => {
    setMessages([]);
    setProcessedEventsTimeline([]);
    setHistoricalActivities({});
    setError(null);
    setCurrentStreamingMessage("");
    setStreamingMetadata({});
    setIsLoading(false);
    // Create a new thread for fresh context
    setCurrentThreadId(null);
    setConversationHistory([]);
    
    // Update URL to remove thread_id parameter
    const url = new URL(window.location.href);
    url.searchParams.delete('thread_id');
    window.history.replaceState({}, '', url.toString());
  }, []);

  const handleLoadHistory = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/conversation-history/default`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const history = await response.json();
        if (history && history.length > 0) {
          setConversationHistory(history);
          
          const historyMessages: Message[] = history.map((entry: any, index: number) => ({
            type: "ai",
            content: entry.final_answer || entry.user_query || "Previous conversation",
            id: entry._id || `history_${Date.now()}_${index}_${Math.random().toString(36).substr(2, 9)}`,
            metadata: {
              processing_time: entry.processing_time,
              query_type: entry.query_type,
              domain_analysis: entry.state_snapshot?.domain_expert_analysis,
              ux_analysis: entry.state_snapshot?.ux_ui_specialist_analysis,
              technical_analysis: entry.state_snapshot?.technical_architect_analysis,
              revenue_analysis: entry.state_snapshot?.revenue_model_analyst_analysis,
              moderator_aggregation: entry.state_snapshot?.moderator_aggregation,
            },
          }));
          
          setMessages(historyMessages);
          console.log("Refreshed conversation history:", historyMessages.length, "messages");
        }
      }
    } catch (error) {
      console.warn("Could not load conversation history:", error);
    }
  }, [apiUrl]);

  // Function to load specific thread context
  const handleLoadThreadContext = useCallback(async (threadId: string) => {
    try {
      const hasContext = await checkAndLoadThreadContext(threadId);
      if (hasContext) {
        console.log(`Successfully loaded context for thread: ${threadId}`);
      } else {
        console.log(`No context found for thread: ${threadId}`);
      }
      return hasContext;
    } catch (error) {
      console.warn("Could not load thread context:", error);
      return false;
    }
  }, [checkAndLoadThreadContext]);

  // Check for thread ID in URL on app start
  useEffect(() => {
    const urlThreadId = getThreadIdFromUrl();
    if (urlThreadId) {
      console.log(`Found thread ID in URL: ${urlThreadId}`);
      // Load context for this thread
      checkAndLoadThreadContext(urlThreadId);
    }
  }, [getThreadIdFromUrl, checkAndLoadThreadContext]);

  // Load conversation history when thread ID changes, but only if no messages exist
  useEffect(() => {
    if (currentThreadId && messages.length === 0) {
      loadConversationHistory(currentThreadId, true);
    }
  }, [currentThreadId, messages.length, loadConversationHistory]);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="h-full w-full max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={isLoading}
              onCancel={handleCancel}
              onNewAnalysis={handleNewAnalysis}
              onLoadHistory={handleLoadHistory}
            />
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="flex flex-col items-center justify-center gap-4">
                <h1 className="text-2xl text-red-400 font-bold">Error</h1>
                <p className="text-red-400">{error}</p>

                <Button
                  variant="destructive"
                  onClick={() => window.location.reload()}
                >
                  Retry
                </Button>
              </div>
            </div>
          ) : (
            <ChatMessagesView
              messages={messages}
              isLoading={isLoading}
              scrollAreaRef={scrollAreaRef}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
              onNewAnalysis={handleNewAnalysis}
              liveActivityEvents={processedEventsTimeline}
              historicalActivities={historicalActivities}
              streamingMessage={currentStreamingMessage}
            />
          )}
          
          {/* Debug info - remove in production */}
          {process.env.NODE_ENV === 'development' && currentThreadId && (
            <div className="fixed bottom-4 right-4 bg-neutral-700 p-2 rounded text-xs opacity-50">
              Thread: {currentThreadId.substring(0, 20)}...
            </div>
          )}
      </main>
    </div>
  );
}

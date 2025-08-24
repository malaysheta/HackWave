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

  // Load conversation history when thread ID changes
  useEffect(() => {
    if (currentThreadId) {
      loadConversationHistory(currentThreadId);
    }
  }, [currentThreadId]);

  // Load initial conversation history on app start
  useEffect(() => {
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
            const historyMessages: Message[] = history.map((entry: any) => ({
              type: "ai",
              content: entry.final_answer || entry.user_query || "Previous conversation",
              id: entry._id || Date.now().toString(),
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
  }, [apiUrl]);

  const loadConversationHistory = async (threadId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/conversation-history/${threadId}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const history = await response.json();
        setConversationHistory(history);
        
        // Convert history to messages format
        const historyMessages: Message[] = history.map((entry: any) => ({
          type: "ai",
          content: entry.final_answer || entry.user_query || "Previous conversation",
          id: entry._id || Date.now().toString(),
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
      
      setProcessedEventsTimeline([]);
      setIsLoading(true);
      setError(null);
      setCurrentStreamingMessage("");
      setStreamingMetadata({});

      // Add user message
      const userMessage: Message = {
        type: "human",
        content: submittedInputValue,
        id: Date.now().toString(),
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

        // After completion, reload conversation history to get updated context
        await loadConversationHistory(threadId);

      } catch (err) {
        console.error('Streaming error:', err);
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setIsLoading(false);
      }
    },
    [apiUrl, getOrCreateThreadId, loadConversationHistory]
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
        // Add the final AI message
        const aiMessage: Message = {
          type: "ai",
          content: currentStreamingMessage || "Analysis completed",
          id: (Date.now() + 1).toString(),
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
          
          const historyMessages: Message[] = history.map((entry: any) => ({
            type: "ai",
            content: entry.final_answer || entry.user_query || "Previous conversation",
            id: entry._id || Date.now().toString(),
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
      </main>
    </div>
  );
}

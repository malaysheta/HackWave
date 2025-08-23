import { InputForm } from "./InputForm";

interface WelcomeScreenProps {
  handleSubmit: (
    submittedInputValue: string,
    effort: string
  ) => void;
  onCancel: () => void;
  onNewAnalysis: () => void;
  isLoading: boolean;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  handleSubmit,
  onCancel,
  onNewAnalysis,
  isLoading,
}) => (
  <div className="h-full flex flex-col items-center justify-center text-center px-4 flex-1 w-full max-w-3xl mx-auto gap-4">
    <div>
      <h1 className="text-5xl md:text-6xl font-semibold text-neutral-100 mb-3">
        Multi-Agent AI
      </h1>
      <p className="text-xl md:text-2xl text-neutral-400 mb-4">
        Product Requirements Refinement System
      </p>
      <p className="text-lg text-neutral-300 mb-6">
        Get comprehensive product requirements analysis from domain experts, UX/UI specialists, and technical architects.
      </p>
    </div>
    
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 w-full max-w-4xl">
      <div className="bg-neutral-700 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-blue-400 mb-2">Domain Expert</h3>
        <p className="text-sm text-neutral-300">Business logic, industry standards, compliance requirements</p>
      </div>
      <div className="bg-neutral-700 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-green-400 mb-2">UX/UI Specialist</h3>
        <p className="text-sm text-neutral-300">User experience, interface design, accessibility</p>
      </div>
      <div className="bg-neutral-700 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-purple-400 mb-2">Technical Architect</h3>
        <p className="text-sm text-neutral-300">System architecture, scalability, implementation</p>
      </div>
    </div>
    
    <div className="w-full mt-4">
      <InputForm
        onSubmit={handleSubmit}
        isLoading={isLoading}
        onCancel={onCancel}
        onNewAnalysis={onNewAnalysis}
        hasHistory={false}
      />
    </div>
    <p className="text-xs text-neutral-500">
      Powered by Google Gemini 2.0 Flash. Debate resolution in under 2 minutes.
    </p>
  </div>
);

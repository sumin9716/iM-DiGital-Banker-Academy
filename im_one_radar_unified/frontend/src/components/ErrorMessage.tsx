import { HiExclamationCircle } from 'react-icons/hi';

interface ErrorMessageProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export default function ErrorMessage({
  title = '오류가 발생했습니다',
  message,
  onRetry,
}: ErrorMessageProps) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] text-center">
      <div className="p-4 bg-red-100 rounded-full">
        <HiExclamationCircle className="w-12 h-12 text-red-500" />
      </div>
      <h3 className="mt-4 text-lg font-semibold text-gray-800">{title}</h3>
      <p className="mt-2 text-gray-500 max-w-md">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="mt-4 btn btn-primary">
          다시 시도
        </button>
      )}
    </div>
  );
}

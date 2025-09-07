import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

export const useSocket = (serverUrl: string): Socket | null => {
  const [socket, setSocket] = useState<Socket | null>(null);

  useEffect(() => {
    const socketConnection = io(serverUrl, {
      transports: ['websocket', 'polling'],
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socketConnection.on('connect', () => {
      console.log('Connected to server:', socketConnection.id);
    });

    socketConnection.on('disconnect', (reason) => {
      console.log('Disconnected from server:', reason);
    });

    socketConnection.on('connect_error', (error) => {
      console.error('Connection error:', error);
    });

    setSocket(socketConnection);

    return () => {
      socketConnection.close();
    };
  }, [serverUrl]);

  return socket;
};
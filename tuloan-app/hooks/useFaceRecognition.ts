import { CameraView, useCameraPermissions } from 'expo-camera';
import { useRef, useState } from "react";
import axios from 'axios';
import * as ImagePicker from 'expo-image-picker';

const ip_url = process.env.EXPO_PUBLIC_IP_URL;

export const useFaceRecog = () => {
    const cameraRef = useRef<CameraView>(null);
    const [permission, requestPermission] = useCameraPermissions();
    const [idUploaded, setIdUploaded] = useState<boolean>(false);
    const [uploadingId, setUploadingId] = useState<boolean>(false);
    const [isComparing, setIsComparing] = useState<boolean>(false);
    const [isMatched, setIsMatched] = useState<boolean>(false);
    const [confidence, setConfidence] = useState<number>(0);
    
    // Liveness detection states
    const [blinkDetected, setBlinkDetected] = useState<boolean>(false);
    const [leftPoseDetected, setLeftPoseDetected] = useState<boolean>(false);
    const [rightPoseDetected, setRightPoseDetected] = useState<boolean>(false);
    
    const isCapturingBlink = useRef<boolean>(false);
    const isCapturingLeft = useRef<boolean>(false);
    const isCapturingRight = useRef<boolean>(false);
    const isCapturingCompare = useRef<boolean>(false);
    const isMounted = useRef<boolean>(true);
    
    const sessionId = useRef(`session-${Date.now()}`).current;

    // Computed: liveness fully verified
    const livenessVerified = blinkDetected && leftPoseDetected && rightPoseDetected;

    const uploadId = async () => {
        try {
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ['images'],
                allowsEditing: true,
                quality: 0.8,
                aspect: [3, 4],
            });

            if (result.canceled || !result.assets?.length) return;

            setUploadingId(true);
            const formData = new FormData();

            const uri = result.assets[0].uri;
            const filename = uri.split('/').pop() || 'id.jpg';
            const match = /\.(\w+)$/.exec(filename);
            const type = match ? `image/${match[1]}` : 'image/jpeg';

            formData.append('file', {
                uri: uri,
                name: filename,
                type: type,
            } as any);

            const response = await axios.post(`${ip_url}/api/facial/v1/upload-id`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            const data = response.data;

            if (data.status === 'success') {
                setIdUploaded(true);
            }
        } catch (err) {
            console.error('Upload ID Error:', err);
        } finally {
            setUploadingId(false);
        }
    };

    const detectBlink = async () => {
        if (!cameraRef.current || blinkDetected || isCapturingBlink.current || !isMounted.current) {
            return;
        }

        isCapturingBlink.current = true;

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.5,
                skipProcessing: true,
            });

            if (!photo?.uri || !isMounted.current) {
                isCapturingBlink.current = false;
                return;
            }

            const formData = new FormData();
            const filename = photo.uri.split('/').pop() || 'blink.jpg';
            const match = /\.(\w+)$/.exec(filename);
            const type = match ? `image/${match[1]}` : 'image/jpeg';

            formData.append('file', {
                uri: photo.uri,
                name: filename,
                type: type,
            } as any);

            formData.append('session_id', sessionId);

            const response = await axios.post(`${ip_url}/api/facial/v1/detect-blink`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                timeout: 5000,
            });

            if (!isMounted.current) return;

            const data = response.data;

            if (data.face_detected && data.blink_detected) {
                setBlinkDetected(true);
            }
        } catch (err: any) {
            if (isMounted.current) {
                console.error('Blink Detection Error:', err);
            }
        } finally {
            isCapturingBlink.current = false;
        }
    };

    const detectHeadTurn = async (direction: 'left' | 'right') => {
        const isCapturingRef = direction === 'left' ? isCapturingLeft : isCapturingRight;
        const isDetected = direction === 'left' ? leftPoseDetected : rightPoseDetected;
        
        if (!cameraRef.current || isDetected || isCapturingRef.current || !isMounted.current) {
            return;
        }

        isCapturingRef.current = true;

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.5,
                skipProcessing: true,
            });

            if (!photo?.uri || !isMounted.current) {
                isCapturingRef.current = false;
                return;
            }

            const formData = new FormData();
            const filename = photo.uri.split('/').pop() || `${direction}.jpg`;
            const match = /\.(\w+)$/.exec(filename);
            const type = match ? `image/${match[1]}` : 'image/jpeg';

            formData.append('file', {
                uri: photo.uri,
                name: filename,
                type: type,
            } as any);

            formData.append('session_id', sessionId);
            formData.append('direction', direction);

            const response = await axios.post(
                `${ip_url}/api/facial/v1/detect-head-turn?direction=${direction}`, 
                formData, 
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                    timeout: 5000,
                }
            );

            if (!isMounted.current) return;

            const data = response.data;

            if (data.face_detected && data.pose_detected) {
                if (direction === 'left') {
                    setLeftPoseDetected(true);
                } else {
                    setRightPoseDetected(true);
                }
            }
        } catch (err: any) {
            if (isMounted.current) {
                console.error(`${direction} Turn Detection Error:`, err);
            }
        } finally {
            isCapturingRef.current = false;
        }
    };

    const captureAndCompare = async () => {
        if (!cameraRef.current || isComparing || !livenessVerified || isCapturingCompare.current || !isMounted.current) {
            return;
        }

        setIsComparing(true);
        isCapturingCompare.current = true;

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.6,
                skipProcessing: true,
            });

            if (!photo?.uri || !isMounted.current) {
                setIsComparing(false);
                isCapturingCompare.current = false;
                return;
            }

            const formData = new FormData();
            const filename = photo.uri.split('/').pop() || 'live.jpg';
            const match = /\.(\w+)$/.exec(filename);
            const type = match ? `image/${match[1]}` : 'image/jpeg';

            formData.append('file', {
                uri: photo.uri,
                name: filename,
                type: type,
            } as any);

            const response = await axios.post(`${ip_url}/api/facial/v1/compare`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                timeout: 10000,
            });

            if (!isMounted.current) return;

            const data = response.data;
            
            if (data.match) {
                setIsMatched(true);
                const confidencePercent = Math.max(0, Math.min(100, (1 - data.distance / data.threshold) * 100));
                setConfidence(confidencePercent);
            } else {
                setIsMatched(false);
                setConfidence(0);
            }
        } catch (err: any) {
            if (isMounted.current) {
                console.error('Compare Error:', err);
                setIsMatched(false);
            }
        } finally {
            if (isMounted.current) {
                setIsComparing(false);
            }
            isCapturingCompare.current = false;
        }
    };

    const resetVerification = async () => {
        isCapturingBlink.current = false;
        isCapturingLeft.current = false;
        isCapturingRight.current = false;
        isCapturingCompare.current = false;
        
        setIdUploaded(false);
        setIsMatched(false);
        setConfidence(0);
        setBlinkDetected(false);
        setLeftPoseDetected(false);
        setRightPoseDetected(false);
        setIsComparing(false);
        
        try {
            await axios.post(`${ip_url}/api/facial/v1/reset-liveness-session`, { session_id: sessionId });
            await axios.post(`${ip_url}/api/facial/v1/reset`);
        } catch (err) {
            console.error('Reset error:', err);
        }
    };

    return {
        cameraRef,
        permission,
        requestPermission,
        idUploaded,
        setIdUploaded,
        uploadingId,
        setUploadingId,
        isComparing,
        setIsComparing,
        isMatched,
        setIsMatched,
        confidence,
        setConfidence,
        blinkDetected,
        setBlinkDetected,
        leftPoseDetected,
        setLeftPoseDetected,
        rightPoseDetected,
        setRightPoseDetected,
        livenessVerified,
        sessionId,
        uploadId,
        detectBlink,
        detectHeadTurn,
        captureAndCompare,
        resetVerification,
        isMounted,
        isCapturingBlink,
        isCapturingLeft,
        isCapturingRight,
        isCapturingCompare
    }
}
import React, { useRef, useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, Animated } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { Audio } from 'expo-av';
import axios from 'axios';

export default function LiveFaceVerification() {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [idUploaded, setIdUploaded] = useState<boolean>(false);
  const [uploadingId, setUploadingId] = useState<boolean>(false);
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [isMatched, setIsMatched] = useState<boolean>(false);
  const [confidence, setConfidence] = useState<number>(0);
  
  // Blink detection state
  const [blinkCount, setBlinkCount] = useState<number>(0);
  const [livenessVerified, setLivenessVerified] = useState<boolean>(false);
  const REQUIRED_BLINKS = 2;
  
  // Prevent simultaneous captures
  const isCapturingBlink = useRef<boolean>(false);
  const isCapturingCompare = useRef<boolean>(false);
  const isMounted = useRef<boolean>(true);
  
  // Animations
  const scanAnimation = useRef(new Animated.Value(0)).current;
  const pulseAnimation = useRef(new Animated.Value(1)).current;
  const borderAnimation = useRef(new Animated.Value(0)).current;
  const progressAnimation = useRef(new Animated.Value(0)).current;
  const successScale = useRef(new Animated.Value(0)).current;
  const fadeIn = useRef(new Animated.Value(0)).current;

  const serverUrl = "http://192.168.1.12:8000/api/facial/v1";
  const sessionId = useRef(`session-${Date.now()}`).current;

  useEffect(() => {
    isMounted.current = true;
    if (!permission?.granted) {
      requestPermission();
    }
    
    return () => {
      isMounted.current = false;
      isCapturingBlink.current = false;
      isCapturingCompare.current = false;
    };
  }, []);

  // Fade in animation on mount
  useEffect(() => {
    Animated.timing(fadeIn, {
      toValue: 1,
      duration: 400,
      useNativeDriver: true,
    }).start();
  }, []);

  // Continuous scanning line animation
  useEffect(() => {
    if (idUploaded) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(scanAnimation, {
            toValue: 1,
            duration: 2000,
            useNativeDriver: true,
          }),
          Animated.timing(scanAnimation, {
            toValue: 0,
            duration: 0,
            useNativeDriver: true,
          }),
        ])
      ).start();
    }
  }, [idUploaded]);

  // Pulse animation for active scanning
  useEffect(() => {
    if (idUploaded && !isMatched) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnimation, {
            toValue: 1.05,
            duration: 1500,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnimation, {
            toValue: 1,
            duration: 1500,
            useNativeDriver: true,
          }),
        ])
      ).start();
    }
  }, [idUploaded, isMatched]);

  // Border color animation
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(borderAnimation, {
          toValue: 1,
          duration: 1500,
          useNativeDriver: false,
        }),
        Animated.timing(borderAnimation, {
          toValue: 0,
          duration: 1500,
          useNativeDriver: false,
        }),
      ])
    ).start();
  }, []);

  // Animate progress bar
  useEffect(() => {
    Animated.timing(progressAnimation, {
      toValue: blinkCount / REQUIRED_BLINKS,
      duration: 300,
      useNativeDriver: false,
    }).start();
  }, [blinkCount]);

  // Success animation
  useEffect(() => {
    if (isMatched) {
      Animated.spring(successScale, {
        toValue: 1,
        tension: 50,
        friction: 7,
        useNativeDriver: true,
      }).start();
    } else {
      successScale.setValue(0);
    }
  }, [isMatched]);

  // Blink detection interval
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (idUploaded && cameraRef.current && !livenessVerified && isMounted.current) {
      interval = setInterval(() => {
        if (!isCapturingBlink.current && isMounted.current) {
          detectBlink();
        }
      }, 500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [idUploaded, livenessVerified]);

  // Face comparison interval (only after liveness verified)
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (idUploaded && livenessVerified && cameraRef.current && !isComparing && isMounted.current) {
       interval = setInterval(() => {
        if (!isCapturingCompare.current && isMounted.current) {
          captureAndCompare();
        }
      }, 1500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [idUploaded, livenessVerified, isComparing]);

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

      const response = await axios.post(`${serverUrl}/upload-id`, formData, {
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
    if (!cameraRef.current || livenessVerified || isCapturingBlink.current || !isMounted.current) {
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

      const response = await axios.post(`${serverUrl}/detect-blink`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 5000,
      });

      if (!isMounted.current) return;

      const data = response.data;
      
      if (data.face_detected) {
        setBlinkCount(data.blink_count);

        if (data.blink_count >= REQUIRED_BLINKS && !livenessVerified) {
          setLivenessVerified(true);
        }
      }
    } catch (err: any) {
      if (isMounted.current) {
        console.error('Blink Detection Error:', err);
      }
    } finally {
      isCapturingBlink.current = false;
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

      const response = await axios.post(`${serverUrl}/compare`, formData, {
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
    isCapturingCompare.current = false;
    
    setIdUploaded(false);
    setIsMatched(false);
    setConfidence(0);
    setBlinkCount(0);
    setLivenessVerified(false);
    setIsComparing(false);
    
    progressAnimation.setValue(0);
    
    try {
      await axios.post(`${serverUrl}/reset-blink-session`, { session_id: sessionId });
      await axios.post(`${serverUrl}/reset`);
    } catch (err) {
      console.error('Reset error:', err);
    }
  };

  if (!permission) {
    return (
      <View className="flex-1 justify-center items-center bg-blue-600">
        <ActivityIndicator size="large" color="#FFFFFF" />
        <Text className="text-white text-lg mt-4 font-medium">Initializing camera...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View className="flex-1 justify-center items-center bg-blue-600 px-8">
        <Animated.View style={{ opacity: fadeIn }} className="items-center">
          <View className="bg-white rounded-full w-24 h-24 items-center justify-center mb-8">
            <Text className="text-6xl">📷</Text>
          </View>
          <Text className="text-white text-3xl font-bold mb-3 text-center">
            Camera Access
          </Text>
          <Text className="text-white/90 text-base mb-10 text-center leading-6">
            We need your permission to access the camera for face verification
          </Text>
          <TouchableOpacity
            onPress={requestPermission}
            className="bg-white px-12 py-4 rounded-full active:bg-gray-100 shadow-lg"
          >
            <Text className="text-blue-600 font-bold text-lg">Allow Camera</Text>
          </TouchableOpacity>
        </Animated.View>
      </View>
    );
  }

  const scanTranslateY = scanAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 300],
  });

  const progressWidth = progressAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  const borderColor = borderAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: isMatched 
      ? ['#10B981', '#34D399']
      : livenessVerified
      ? ['#3B82F6', '#60A5FA']
      : ['#0066FF', '#4D94FF'],
  });

  return (
    <View className="flex-1 bg-white">
      {!idUploaded ? (
        <Animated.View style={{ opacity: fadeIn }} className="flex-1 justify-center items-center bg-blue-600 px-8">
          <View className="items-center">
            <View className="bg-white rounded-full w-32 h-32 items-center justify-center mb-8 shadow-xl">
              <Text className="text-7xl">🆔</Text>
            </View>
            <Text className="text-white text-3xl font-bold mb-4 text-center">
              Face Verification
            </Text>
            <Text className="text-white/90 text-base mb-12 text-center leading-6">
              Upload a photo of your valid ID to begin the verification process
            </Text>
            <TouchableOpacity
              onPress={uploadId}
              disabled={uploadingId}
              className={`w-full px-10 py-5 rounded-full shadow-xl ${
                uploadingId ? 'bg-white/50' : 'bg-white active:bg-gray-100'
              }`}
            >
              {uploadingId ? (
                <View className="flex-row items-center justify-center">
                  <ActivityIndicator size="small" color="#0066FF" />
                  <Text className="text-blue-600 font-bold text-lg ml-3">Uploading...</Text>
                </View>
              ) : (
                <Text className="text-blue-600 font-bold text-lg text-center">
                  Upload ID Photo
                </Text>
              )}
            </TouchableOpacity>
          </View>
        </Animated.View>
      ) : (
        <View className="flex-1 bg-white">
          {/* Header */}
          <View className="pt-14 pb-6 px-6 bg-blue-600">
            <Text className="text-white text-2xl font-bold text-center mb-1">
              Face Verification
            </Text>
            <Text className="text-white/80 text-sm text-center">
              {!livenessVerified 
                ? `Blink ${REQUIRED_BLINKS} times naturally`
                : isMatched
                ? "Verification successful!"
                : "Verifying your identity..."
              }
            </Text>
          </View>

          {/* Main Content */}
          <View className="flex-1 px-6 pt-8">
            {/* Progress Section */}
            {!livenessVerified && (
              <View className="bg-blue-50 rounded-3xl p-5 mb-6">
                <View className="flex-row items-center justify-between mb-3">
                  <Text className="text-blue-900 font-semibold text-base">
                    Liveness Check
                  </Text>
                  <Text className="text-blue-600 font-bold text-base">
                    {blinkCount}/{REQUIRED_BLINKS}
                  </Text>
                </View>
                <View className="h-2 bg-blue-200 rounded-full overflow-hidden">
                  <Animated.View 
                    style={{ 
                      width: progressWidth,
                      backgroundColor: '#0066FF',
                      height: '100%',
                    }} 
                  />
                </View>
                <Text className="text-blue-700 text-xs mt-3 text-center">
                  Blink naturally to confirm you're a real person
                </Text>
              </View>
            )}

            {/* Camera Preview */}
            <View className="flex-1 justify-center items-center">
              <View className="relative" style={{ width: 300, height: 300 }}>
                {/* Glow Effect */}
                <Animated.View
                  style={{
                    position: 'absolute',
                    top: -15,
                    left: -15,
                    width: 330,
                    height: 330,
                    borderRadius: 165,
                    backgroundColor: isMatched 
                      ? 'rgba(16, 185, 129, 0.15)'
                      : 'rgba(0, 102, 255, 0.15)',
                    transform: [{ scale: pulseAnimation }],
                  }}
                />

                {/* Camera */}
                <View
                  style={{
                    width: 300,
                    height: 300,
                    borderRadius: 150,
                    overflow: 'hidden',
                    backgroundColor: '#000',
                  }}
                >
                  <CameraView 
                    ref={cameraRef} 
                    style={{ width: 300, height: 300 }}
                    facing="front"
                    mirror={true}
                  />
                </View>

                {/* Border */}
                <Animated.View 
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: 300,
                    height: 300,
                    borderRadius: 150,
                    borderWidth: 5,
                    borderColor: borderColor,
                  }}
                />

                {/* Scanning Line */}
                <Animated.View 
                  style={{ 
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    width: 300,
                    height: 3,
                    backgroundColor: isMatched ? '#10B981' : '#0066FF',
                    transform: [{ translateY: scanTranslateY }],
                    shadowColor: isMatched ? '#10B981' : '#0066FF',
                    shadowOpacity: 0.8,
                    shadowRadius: 10,
                  }} 
                />

                {/* Corner Guides */}
                {[0, 90, 180, 270].map((rotation, index) => (
                  <View
                    key={index}
                    style={{
                      position: 'absolute',
                      top: 5,
                      left: 5,
                      width: 30,
                      height: 30,
                      borderTopWidth: 4,
                      borderLeftWidth: 4,
                      borderColor: isMatched ? '#10B981' : '#0066FF',
                      borderTopLeftRadius: 8,
                      transform: [
                        { rotate: `${rotation}deg` },
                        { translateX: rotation === 90 || rotation === 180 ? 260 : 0 },
                        { translateY: rotation === 180 || rotation === 270 ? 260 : 0 },
                      ],
                    }}
                  />
                ))}

                {/* Liveness Badge */}
                {livenessVerified && (
                  <View style={{
                    position: 'absolute',
                    top: -10,
                    left: -10,
                    backgroundColor: '#10B981',
                    borderRadius: 25,
                    width: 50,
                    height: 50,
                    alignItems: 'center',
                    justifyContent: 'center',
                    shadowColor: '#10B981',
                    shadowOpacity: 0.5,
                    shadowRadius: 10,
                  }}>
                    <Text style={{ color: 'white', fontSize: 24, fontWeight: 'bold' }}>✓</Text>
                  </View>
                )}

                {/* Success Badge */}
                {isMatched && (
                  <Animated.View style={{
                    position: 'absolute',
                    top: -10,
                    right: -10,
                    backgroundColor: '#10B981',
                    borderRadius: 25,
                    width: 50,
                    height: 50,
                    alignItems: 'center',
                    justifyContent: 'center',
                    shadowColor: '#10B981',
                    shadowOpacity: 0.5,
                    shadowRadius: 10,
                    transform: [{ scale: successScale }],
                  }}>
                    <Text style={{ color: 'white', fontSize: 24, fontWeight: 'bold' }}>✓</Text>
                  </Animated.View>
                )}
              </View>
            </View>

            {/* Success Card */}
            {isMatched && (
              <Animated.View 
                style={{ transform: [{ scale: successScale }] }}
                className="bg-green-50 rounded-3xl p-6 mb-6 border-2 border-green-200"
              >
                <Text className="text-green-700 text-center text-xl font-bold mb-4">
                  ✓ Identity Verified
                </Text>
                <View className="flex-row justify-around">
                  <View className="items-center">
                    <Text className="text-green-600 text-xs mb-1">Liveness</Text>
                    <View className="bg-green-100 rounded-full w-12 h-12 items-center justify-center">
                      <Text className="text-green-700 text-xl font-bold">✓</Text>
                    </View>
                  </View>
                  <View className="items-center">
                    <Text className="text-green-600 text-xs mb-1">Match Score</Text>
                    <View className="bg-green-100 rounded-full w-12 h-12 items-center justify-center">
                      <Text className="text-green-700 text-sm font-bold">{confidence.toFixed(0)}%</Text>
                    </View>
                  </View>
                </View>
              </Animated.View>
            )}
          </View>

          {/* Bottom Actions */}
          <View className="pb-10 px-6 bg-white">
            <View className="flex-row gap-3">
              <TouchableOpacity
                onPress={resetVerification}
                className="flex-1 bg-gray-100 px-6 py-4 rounded-full active:bg-gray-200"
              >
                <Text className="text-gray-700 text-center text-base font-bold">
                  Start Over
                </Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                disabled={!isMatched}
                className={`flex-1 px-6 py-4 rounded-full ${
                  isMatched ? 'bg-blue-600 active:bg-blue-700' : 'bg-gray-300'
                }`}
              >
                <Text className={`text-center text-base font-bold ${
                  isMatched ? 'text-white' : 'text-gray-500'
                }`}>
                  {isMatched ? 'Continue' : 'Processing...'}
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      )}
    </View>
  );
}
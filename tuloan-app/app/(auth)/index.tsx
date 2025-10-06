import React, { useRef, useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, Animated } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { useFaceRecog } from '@/hooks/useFaceRecognition';

export default function LiveFaceVerification() {
  const {
    cameraRef,
    permission,
    requestPermission,
    idUploaded,
    uploadingId,
    isComparing,
    isMatched,
    confidence,
    blinkDetected,
    leftPoseDetected,
    rightPoseDetected,
    livenessVerified,
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
  } = useFaceRecog();

  const scanAnimation = useRef(new Animated.Value(0)).current;
  const pulseAnimation = useRef(new Animated.Value(1)).current;
  const borderAnimation = useRef(new Animated.Value(0)).current;
  const progressAnimation = useRef(new Animated.Value(0)).current;
  const successScale = useRef(new Animated.Value(0)).current;
  const fadeIn = useRef(new Animated.Value(0)).current;

  // Current step: 'blink', 'left', 'right', or 'verify'
  const [currentStep, setCurrentStep] = useState<'blink' | 'left' | 'right' | 'verify'>('blink');

  useEffect(() => {
    isMounted.current = true;
    if (!permission?.granted) {
      requestPermission();
    }

    return () => {
      isMounted.current = false;
      isCapturingBlink.current = false;
      isCapturingLeft.current = false;
      isCapturingRight.current = false;
      isCapturingCompare.current = false;
    };
  }, []);

  // Fade in animation
  useEffect(() => {
    Animated.timing(fadeIn, {
      toValue: 1,
      duration: 400,
      useNativeDriver: true,
    }).start();
  }, []);

  // Scanning line animation
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

  // Pulse animation
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

  // Border animation
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

  // Progress animation
  useEffect(() => {
    const totalSteps = 3;
    let completedSteps = 0;
    if (blinkDetected) completedSteps++;
    if (leftPoseDetected) completedSteps++;
    if (rightPoseDetected) completedSteps++;

    Animated.timing(progressAnimation, {
      toValue: completedSteps / totalSteps,
      duration: 300,
      useNativeDriver: false,
    }).start();
  }, [blinkDetected, leftPoseDetected, rightPoseDetected]);

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

  // Update current step based on completion
  useEffect(() => {
    if (!blinkDetected) {
      setCurrentStep('blink');
    } else if (!leftPoseDetected) {
      setCurrentStep('left');
    } else if (!rightPoseDetected) {
      setCurrentStep('right');
    } else {
      setCurrentStep('verify');
    }
  }, [blinkDetected, leftPoseDetected, rightPoseDetected]);

  // Blink detection interval
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (idUploaded && currentStep === 'blink' && !blinkDetected && isMounted.current) {
      interval = setInterval(() => {
        if (!isCapturingBlink.current && isMounted.current) {
          detectBlink();
        }
      }, 500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [idUploaded, currentStep, blinkDetected]);

  // Left turn detection interval
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (idUploaded && currentStep === 'left' && !leftPoseDetected && isMounted.current) {
      interval = setInterval(() => {
        if (!isCapturingLeft.current && isMounted.current) {
          detectHeadTurn('left');
        }
      }, 500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [idUploaded, currentStep, leftPoseDetected]);

  // Right turn detection interval
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (idUploaded && currentStep === 'right' && !rightPoseDetected && isMounted.current) {
      interval = setInterval(() => {
        if (!isCapturingRight.current && isMounted.current) {
          detectHeadTurn('right');
        }
      }, 500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [idUploaded, currentStep, rightPoseDetected]);

  // Face comparison interval (after all liveness checks)
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (idUploaded && livenessVerified && !isComparing && isMounted.current) {
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
        <Animated.View style={{ opacity: fadeIn }} className="flex-1 bg-blue-600">
          <View className="flex-1 justify-center items-center px-8">
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
          </View>

          <View className="px-8 pb-8">
            <View className="bg-white/10 rounded-3xl p-6 backdrop-blur">
              <Text className="text-white font-semibold text-base mb-4">
                📋 Accepted IDs
              </Text>
              <View className="space-y-3">
                <View className="flex-row items-center">
                  <View className="w-2 h-2 bg-white rounded-full mr-3" />
                  <Text className="text-white/90 text-sm">Driver's License</Text>
                </View>
                <View className="flex-row items-center">
                  <View className="w-2 h-2 bg-white rounded-full mr-3" />
                  <Text className="text-white/90 text-sm">Passport</Text>
                </View>
                <View className="flex-row items-center">
                  <View className="w-2 h-2 bg-white rounded-full mr-3" />
                  <Text className="text-white/90 text-sm">National ID</Text>
                </View>
                <View className="flex-row items-center">
                  <View className="w-2 h-2 bg-white rounded-full mr-3" />
                  <Text className="text-white/90 text-sm">Postal ID</Text>
                </View>
              </View>
            </View>

            <View className="bg-white/10 rounded-3xl p-6 mt-4 backdrop-blur">
              <Text className="text-white font-semibold text-base mb-3">
                💡 Tips for best results
              </Text>
              <Text className="text-white/80 text-sm leading-5">
                • Ensure your ID is well-lit and in focus{'\n'}
                • All corners should be visible{'\n'}
                • Avoid glare and shadows
              </Text>
            </View>
          </View>

          <View className="items-center pb-8">
            <View className="flex-row items-center bg-white/10 px-4 py-2 rounded-full">
              <Text className="text-white/70 text-xs mr-1">🔒</Text>
              <Text className="text-white/70 text-xs">Your data is encrypted and secure</Text>
            </View>
          </View>
        </Animated.View>
      ) : (
        <View className="flex-1 bg-white">
          {/* Header */}
          <View className="pt-14 pb-6 px-6 bg-blue-600">
            <Text className="text-white text-2xl font-bold text-center mb-1">
              Face Verification
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
                    {[blinkDetected, leftPoseDetected, rightPoseDetected].filter(Boolean).length}/3
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

                {/* Step indicators */}
                <View className="flex-row justify-between mt-4">
                    {!blinkDetected && !leftPoseDetected && !rightPoseDetected ? (
                      <Text className="text-blue-600 font-medium text-sm">Please blink naturally to confirm you're a live user.</Text>
                    ) : (null)}

                    {blinkDetected && !leftPoseDetected && !rightPoseDetected ? (
                      <Text className="text-blue-600 font-medium text-sm">Please show the left side of your face hold for a moment.</Text>
                    ) : (null)}

                    {blinkDetected && leftPoseDetected && !rightPoseDetected ? (
                      <Text className="text-blue-600 font-medium text-sm">Please show the right side of your face hold for a moment.</Text>
                    ) : (null)}


                </View>
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

                {/* Current Step Indicator */}
                

                {/* Liveness Badge */}
                {livenessVerified && (
                  <View
                    style={{
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
                    }}
                  >
                    <Text style={{ color: 'white', fontSize: 24, fontWeight: 'bold' }}>✓</Text>
                  </View>
                )}

                {/* Success Badge */}
                {isMatched && (
                  <Animated.View
                    style={{
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
                    }}
                  >
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
                <Text className="text-gray-700 text-center text-base font-bold">Start Over</Text>
              </TouchableOpacity>

              <TouchableOpacity
                disabled={!isMatched}
                className={`flex-1 px-6 py-4 rounded-full ${
                  isMatched ? 'bg-blue-600 active:bg-blue-700' : 'bg-gray-300'
                }`}
              >
                <Text
                  className={`text-center text-base font-bold ${
                    isMatched ? 'text-white' : 'text-gray-500'
                  }`}
                >
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
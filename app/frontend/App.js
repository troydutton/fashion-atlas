import CameraView from './CameraView';
import { Image, Button, StyleSheet, Text, TouchableOpacity, View, useWindowDimensions  } from 'react-native';

export default function App() {
  return (
    <View style={{flex: 1, justifyContent: 'center'}}>
      <CameraView />
    </View>
  );
}
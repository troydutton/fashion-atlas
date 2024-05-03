import { StyleSheet, View } from 'react-native';
import CameraView from './CameraView';
import { useEffect } from 'react';

export default function App() {

  useEffect(() => {
    fetch("/members").then(res => res.json()).then(console.log);
  }
  , []);
  
  return (
    <View style={styles.container}>
      <CameraView />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  }
});
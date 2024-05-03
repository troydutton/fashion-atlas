import { StyleSheet, View } from 'react-native';
import CameraView from './CameraView';

export default function App() {
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
// styles/YourStyles.js
import { Image, Button, StyleSheet, Text, TouchableOpacity, View, useWindowDimensions, Dimensions} from 'react-native';

const windowWidth = Dimensions.get('window').width;
const windowHeight = Dimensions.get('window').height;;
const cameraHeight = Math.round((windowWidth * 16) / 9);

// Define your styles here
export const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  permissionTextContainer: {
    marginTop: windowHeight * 0.5,
    justifyContent: 'center', 
    marginBottom: 20,
  },
  text: {
    textAlign: 'center',
    color: 'white',
    fontSize: 20,
  },
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'center', // Center the buttons horizontally
    backgroundColor: 'transparent',
    padding: 20, // Add some padding if needed
  },
  flipButton: {
    alignSelf: 'center',
    position: 'absolute', // Position the flip button absolutely
    left: 40, // Position it a little to the left
  },
  takePictureButton: {
    alignSelf: 'center', // Center the take picture button vertically
  },
  camera: {
    height: cameraHeight,
    marginTop: 45
  },
  image: {
    width: windowWidth * 0.5,
    height: (windowHeight - 40) * 0.5,
  },
});

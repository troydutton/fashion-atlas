import React from 'react';
import { View, Text, Button, StyleSheet, Image } from 'react-native';

const Home = ({ navigation }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Fashion Atlas</Text>
      <Image source={require('./assets/icon.png')} style={styles.logo} />
      <Button
        title="Go to Camera"
        color="#ADD8E6" // Light blue color for the button text
        onPress={() => navigation.navigate('Camera')}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#333333', // Dark grey background
  },
  title: {
    fontSize: 24,
    color: '#ADD8E6', // Light blue color for the text
    marginBottom: 20,
  },
  logo: {
    width: 100,
    height: 100,
    marginBottom: 20,
  },
});

export default Home;

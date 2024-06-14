import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import CameraView from './CameraView';
import SelectionView from './SelectionView';
import ImageView from './ImageView';
import Home from './Home'; // Import the new HomeScreen component

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={Home} options={{ headerShown: false }} />
        <Stack.Screen name="Camera" component={CameraView} options={{ headerShown: false }} />
        {/* <Stack.Screen name="Selection" component={SelectionView} options={{ headerShown: false }}/> */}
        <Stack.Screen name="Images" component={ImageView} options={{ headerShown: false }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

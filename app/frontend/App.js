import CameraView from './CameraView';
import SelectionView from './SelectionView';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { View } from 'react-native';

const Stack = createStackNavigator();

export default function App() {
  return (
    <View style={{flex: 1, justifyContent: 'center'}}>
      <NavigationContainer>
        <Stack.Navigator initialRouteName="Camera">
          <Stack.Screen name="Camera" component={CameraView} options={{ headerShown: false }}/>
          <Stack.Screen name="Selection" component={SelectionView} options={{ headerShown: false }}/>
        </Stack.Navigator>
      </NavigationContainer>
    </View>
  );
}
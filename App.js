import React, { useState, useEffect, useRef } from 'react'
import { StyleSheet, Text, View, Pressable, ImageBackground, Dimensions } from 'react-native'
import { Camera } from 'expo-camera'
import { manipulateAsync, FlipType, SaveFormat } from 'expo-image-manipulator'
import * as tf from '@tensorflow/tfjs'
import * as jpeg from 'jpeg-js'
import { Buffer } from 'buffer'

// Must import & override fetch otherwise will get "FileReader.readAsArrayBuffer is not implemented."
// Causes "Platform browser has already been set. Overwriting the platform with [object Object]."
import { bundleResourceIO } from '@tensorflow/tfjs-react-native'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as cocossd from '@tensorflow-models/coco-ssd'

import modelJson from './nn_model/tfjs_model/model.json'
import modelWeights from './nn_model/tfjs_model/group1-shard1of1.bin'

export default function App() {

  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.front);
	const [cameraReady, setCameraReady] = useState(false)
	const [tfReady, setTfReady] = useState(false)
	const camera = useRef(null)
	const [photo, setPhoto] = useState(null)
	const [sm, setSm] = useState(null)
	const [mm, setMm] = useState(null)
	const [cm, setCm] = useState(null)
	const [sm_pred, setSmPred] = useState([])
	const [mm_pred, setMmPred] = useState([])
	const [cm_pred, setCmPred] = useState(null)
	const [across, setacross] = useState(96)

	// Request permissions
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync()
      setHasPermission(status === 'granted')
    })()
		setPhoto(null)
  }, [])

	useEffect(() => {
		if (hasPermission) {
			let dim = Dimensions.get('window')
			setacross(dim.height > dim.width ? dim.width : dim.height)
		}
	}, [ hasPermission ])

	// Load TensorFlow + models
	useEffect( async () => {
		// tf.setBackend('cpu').then(
		tf.ready().then(async () => {
			setTfReady(true)
			mobilenet.load().then(setSm)
			cocossd.load().then(setMm)
			tf.loadLayersModel(
				bundleResourceIO(
					modelJson, 
					modelWeights
				)
			).then(setCm)
		})
		// )
	}, [])

	useEffect(async () => {
		if (photo && tfReady) {

			let image = jpeg.decode(
				Buffer.from(
					photo.base64,
					// photo.base64.split('base64,')[1], 
					'base64'
				), 
				{ useTArray: true, formatAsRGBA: false }
			)

			let tensor = tf.tensor3d(
				image.data, 
				[image.height, image.width, 3]
			)

			let greyscale = new Uint8Array(image.height * image.width)

			let i = 0
			for (let i = 0; i < image.height * image.width; i++) {
				greyscale[i] = 
					image.data[3 * i + 0] * 0.2989 + 
					image.data[3 * i + 1] * 0.5870 + 
					image.data[3 * i + 2] * 0.1140
			}

			let gs_tensor = tf.tensor4d(
				greyscale,
				[1, image.height, image.width, 1]
			)

			if (sm) {
				sm.classify(tensor).then(setSmPred)
			}

			if (mm) {
				mm.detect(tensor).then(predictions => {
					setMmPred(predictions)

					console.log('mm_prediction made')

					console.log(predictions)
					if (predictions.length && predictions[0].class === 'person') {
						console.log('largest is person')
						// ~80% of 96 + 96
						if (predictions[0].bbox[2] + predictions[0].bbox[3] > 150) { 
							console.log('mostly person')
							if (cm) {
								console.log('facial keypoints detection')
								let predictions = cm.predict(gs_tensor)
								predictions.data().then(setCmPred)
								// cm.predict(gs_tensor).data().then(setCmPred)

							}
						}
					}
				})
			}

		}
	}, [ photo ])

	useEffect(() => {
		console.log(cm_pred)
	}, [ cm_pred ])

	// Deal with situations surrounding no / awaiting permissions
  if (hasPermission === null) {
    return <View />
  }

  if (hasPermission === false) {
    return <View style={styles.middletext}>
			<Text>No access to camera</Text>
		</View>
  }

	// GUI
  return (
    <>
			{tfReady && sm && mm && cm ? 
				<View style={styles.container}>
					<View style={{
						height: 100,
						alignItems: 'center',
						justifyContent: 'center',
					}}>
						<Text style={{color: 'black'}}>title</Text>
					</View>
					
					{photo ? 
						<>
							<Pressable
								onPressOut={() => {
									if (sm_pred.length) {
										setPhoto(null)
										setSmPred([])
										setMmPred([])
										setCmPred(null)
									}
								}}
							>

								<ImageBackground
									style={{ 
										width: across, 
										height: across,
									}}
									source={photo}
								>
									{
										mm_pred.map(e => <Text
												key={e.class + ' ' + e.score}
												style={[ styles.box, { 
													marginLeft: e.bbox[0] / 96 * across,
													marginTop: e.bbox[1] / 96 * across,
													width: e.bbox[2] / 96 * across,
													height: e.bbox[3] / 96 * across,
												}]}
											>
												{e.class + ' ' + e.score}
											</Text>
										)
									}
									{ cm_pred ? 
										<>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[0] / 96 * across,
													marginTop: cm_pred[1] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[2] / 96 * across,
													marginTop: cm_pred[3] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[4] / 96 * across,
													marginTop: cm_pred[5] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[6] / 96 * across,
													marginTop: cm_pred[7] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[8] / 96 * across,
													marginTop: cm_pred[9] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[10] / 96 * across,
													marginTop: cm_pred[11] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[12] / 96 * across,
													marginTop: cm_pred[13] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[14] / 96 * across,
													marginTop: cm_pred[15] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[16] / 96 * across,
													marginTop: cm_pred[17] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[18] / 96 * across,
													marginTop: cm_pred[19] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[20] / 96 * across,
													marginTop: cm_pred[21] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[22] / 96 * across,
													marginTop: cm_pred[23] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[24] / 96 * across,
													marginTop: cm_pred[25] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[26] / 96 * across,
													marginTop: cm_pred[27] / 96 * across,
												}]}
											/>
											<View
												style={[styles.rectangle, {
													marginLeft: cm_pred[28] / 96 * across,
													marginTop: cm_pred[29] / 96 * across,
												}]}
											/>
										</>
									:
										<></>
									}
								</ImageBackground>
							</Pressable>
							<View style={styles.container}>
								{
									sm_pred.map(e => 
										<Text 
											style={{ 
												color: 'black',
												flex: 1,
											}}
											key={e.className}
										>
											{e.className + ' ' + e.probability}
										</Text>
									)
								}
							</View>
						</>
						:
						<>
							<Camera 
								style={{ 
									width: across, 
									height: across,
								}}
								type={type}
								onCameraReady={() => setCameraReady(true)}
								ref={camera}
							/>

							<View style={styles.buttonContainer}>
								<Pressable
									style={styles.button}
									onPressOut={() => {
										setType(
											type === Camera.Constants.Type.back
												? Camera.Constants.Type.front
												: Camera.Constants.Type.back
										);
									}}
								>
									<Text style={styles.text}> Flip </Text>
								</Pressable>

								<Pressable
									style={styles.button}
									onPressOut={async () => {
										if (cameraReady) {
											camera.current.takePictureAsync(
												{quality: 0}
											).then(p => {
												let s = Math.min(p.width, p.height)
												if (type === Camera.Constants.Type.front) {
													manipulateAsync(
														p.uri, 
														[
															{ rotate: 180 }, 
															{ flip: FlipType.Vertical }, 
															{ crop: { 
																	originX: (p.width - s) / 2, 
																	originY: (p.height - s) / 2, 
																	width: s, 
																	height: s
																}
															}, 
															{ resize: { width: 96, height: 96 } }
														],
														{ base64: true, compress: 1, format: SaveFormat.JPEG }
													).then(setPhoto)
												} else {
													manipulateAsync(
														p.uri, 
														[
															{ crop: { 
																	originX: (p.width - s) / 2, 
																	originY: (p.height - s) / 2, 
																	width: s, 
																	height: s
																}
															}, 
															{ resize: { width: 96, height: 96 } }
														],
														{ base64: true, compress: 1, format: SaveFormat.JPEG }
													).then(setPhoto)
												}
											})
										}
									}}
								>
									<Text style={styles.text}> Clip </Text>
								</Pressable>
							</View>
						</>
					}
				</View>
			:
			<View style={styles.middletext}>
				<Text>
					Loading...
				</Text>
			</View>
			}
    </>
  )
}

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
		padding: 50,
  },
  text: {
    fontSize: 18,
    color: 'black',
  },
	middletext: {
		flex: 1,
		alignItems: 'center',
		justifyContent: 'center',
	},
	rectangle: {
		height: 5,
		width: 5,
		backgroundColor: 'red',
		position: 'absolute',
		zIndex: 99,
	},
	box: {
		position: 'absolute',
		borderWidth: 2,
		borderColor: 'red',
		color: 'red',
	}
})
import {launch, goto, checkPopup, evalCode, evalCity, alertClose, getpageLength, getData, writerFile} from './modules/crawler.js'

// main이라는 함수 정의
async function main () {
    await launch('busan', 'haeundae')

    await goto('https://www.pharm114.or.kr/main.asp')

    await checkPopup()

    await evalCode()

    await evalCity()

    await alertClose()

    await getpageLength()

    await getData()

    await writerFile()

    process.exit(1) 
}

// main이라는 함수 실행
main()
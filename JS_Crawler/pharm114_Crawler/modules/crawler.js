import puppeteer from 'puppeteer-core'
import os from 'os'
import fs from 'fs'

const macUrl = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const whidowsUrl = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
const currentOs = os.type()
const launchConfig = {
  headless: false,
  defaultViewport: null,
  ignoreDefaultArgs: ['--disable-extensions'],
  args: [ '--no-sandbox', '--disable-setuid-sandbox', '--disable-notifications', '--disable-extensions'],
  executablePath: currentOs == 'Darwin' ? macUrl : whidowsUrl
}

// 전역변수 (global 변수) : 함수 외부에서 정의 / 하나가 아닌 모든 함수에 엑세스 가능
let browser = null
let page = null
let sido = null
let sigungu = null
let pageLength = 0
let finalData = []   // getData에서 사용

const pageSelector = "body > table:nth-child(2) > tbody > tr > td:nth-child(1) > table > tbody > tr > td:nth-child(2) > table > tbody > tr:nth-child(5) > td > table:nth-child(5) > tbody > tr:nth-child(4) > td > table > tbody > tr > td:nth-child(3)"

// async : function 앞에 위치 / 프라미스 반환
// await(기다리다) : async 함수 안에서만 동작(일반함수에선 사용 불가) / 프라미스가 처리될 때 까지 기다림

// 1) launch 함수 생성 - 빈페이지 열기
// puppeteer을 실행하는 함수, init (puppeteer : Headless Chrome or Chromium 제어하기 위해 도와주는 library)
// puppeteer 기능 : 웹 페이지 crawling 가능
const launch = async function (arg1, arg2) {
  sido = arg1
  sigungu = arg2
  browser = await puppeteer.launch(launchConfig) // 빈 브라우저 실행

  // 지역변수 (함수 내 선언 된 변수)
  const pages = await browser.pages()
  page = pages[0] // 빈 페이지 한개밖에 안열림(default값) page로 전역변수 설정
}

// 2) goto 함수 생성 - 원하는 웹페이지로 이동
const goto = async function (url) {
  return await page.goto(url)
}

// 3) checkPopup 함수 생성 - 팝업창 종료
const checkPopup = async function () {
  const pages = await browser.pages()
  // launch 함수에서 pages 선언해줬다고 안해주면 안된다
  // 이미 launch함수는 끝났으므로 pages 변수는 삭제 돼서 재지정 해줘야됨
  await pages.at(-1).close()
  // [기본창]-[팝업창] 순으로 뜨는데 index [0]-[1]로 나타냄
  // 팝업창 인덱스 1 대신 맨 마지막 [-1]로 쓸 수 있음
  // 사용시 at이라는 문법과 같이 사용
}

// 4) evalCode 함수 생성 - 내가 원하는 지역 클릭
const evalCode = async function () {
  
  await page.evaluate(function (sido) {
    // document.querySelector("#continents > li.seoul > a") : copy JS path
    document.querySelector(`#continents > li.${sido} > a`).click() // 받은 name으로 클릭해주기
  }, sido) 
}

// 5) evalCity 함수 생성 - 내가 원하는 시*군*구 클릭
const evalCity = async function () {

  // 새로운 페이지로 넘어갈 때 페이지가 로드될때까지 대기
  await page.waitForSelector(`#container #continents > li.${sigungu} > a`)

  // 시*군*구 클릭
  await page.evaluate(function (sigungu) {
    document.querySelector(`#container #continents > li.${sigungu} > a`).click()
  }, sigungu)
}

// 6) alertClose 함수 생성 - 알림메세지창(alert창) 닫기
const alertClose = async function () {

  // .on 함수
  // a.on(event:String , function(){})
  return await page.on('dialog', async function(dialog) { // 알림창 닫을땐 dialog(다이얼로그)를 매개변수로 받아 함수를 실행시켜준다
                                                          // 매개변수는 상황에 맞게 정해져있고 그 매개변수를 함수안에 알맞게 넣어줘야한다 
    await dialog.accept() // 알림창은 확인이라는 버튼을 눌러야지 닫히므로 accept라고 써주는것이다
  }) 
}

// 7) getpageLength 함수 생성 - 총 페이지의 수를 세어줌
const getpageLength = async function () {

  // 새로운 페이지로 넘어갔으므로 페이지 로드될 때까지 대기
  // page 넘어가는 버튼 부분이 페이지의 아래쪽에 있으므로 그 지점까지 도달 했을때는 거의 페이지 로딩이 끝났다고 생각 할 수 있어 넣어준것.
  await page.waitForSelector(pageSelector)

  // 해당 지역의 총 쪽수 세어주기
  // pageLength 전역변수로 썼기때문에 const 안적어도됨
  pageLength = await page.evaluate(function (pageSelector) {

    // 부모의 JS path를 불러와 그것들의 아이들(children) = 각각의 쪽수들의 개수를 result로 정의
    const result = document.querySelector(pageSelector).children.length
    return result

  }, pageSelector)

  // console.log('total pages :', pageLength)
}

// 8) getData 함수 생성 - 데이터 불러오기
const getData = async function() {

  // 페이지 수만큼 반복
  // 첫번째 페이지(index=0)는 이미 선택이 되어있으므로 두번째 페이지부터 반복되면 되므로 i = 1(인덱스번호) 부터 해준다.
  for (let i=1; i <= pageLength; i++) {

    // 쪽수 바뀔 때마다 페이지 로드 해줘야됨
    await page.waitForSelector(pageSelector)

    // 약국명 * 주소 * 전화번호 * 운영시간
    const infoArr = await page.evaluate(function(i, sido, sigungu) {
      var trArr = document.querySelectorAll("#printZone > table:nth-child(2) > tbody tr")
                                       // printZone 바로 아래에 있는 table중 두번째 table을 찾겠다 라는 뜻
                                                         //  nth-child : css 언어 (index와 달리 1번부터 시작함)
                                       // 부등호 사용 : 바로 아래 것만 해당, 부등호 안사용 : 아래있는 모든것 뽑겠다

      var returnData = [] // [약국명 * 주소 * 전화번호 * 운영시간] 담을 빈 리스트 생성

      // 데이터 긁기
      for (var i=0; i < trArr.length; i++) {
    
        var currentTr = trArr[i]

        // 약국명
        var name = currentTr.querySelectorAll('td')[1]?.innerText.replaceAll('\n', ' ').replaceAll('\t', ' ') // 출력시 불필요한 문자는 공백으로 대체

        // 주소
        var address = currentTr.querySelectorAll('td')[2]?.innerText.replaceAll('\n', ' ').replaceAll('\t', ' ')

        // 전화번호
        var tel = currentTr.querySelectorAll('td')[3]?.innerText.replaceAll('\n', ' ').replaceAll('\t', ' ')

        // 운영시간
        var open = currentTr.querySelectorAll('td')[4]?.innerText.replaceAll('\n', ' ').replaceAll('\t', ' ')

        // .innerText 만 쓰면 : 값이 존재하면 내용을 불러오지만 값이 존재하지 않을때 에러가 발생한다
        // 따라서 앞에 물음표를 적어주면 
        // 즉, ?.innerText 라고 적을시 존재하는 값들은 불러오고 존재하지 않는것들은 undefined로 반환 시켜서 나오게 해준다 (에러가 발생하지 않음)

        var jsonData = { name, address, tel, open, sido, sigungu }

        // var jsonData = {
        //     'name' : name,
        //     'address' : address,
        //     'tel' : tel,
        //     open
        // }
        // 이런식으로 써도 됨 -> 'open' : open 또는 open 만 써도 가능

        // undefined가 아닌 값들만 returnData에 넣어주기 (push)
        if ( jsonData.address != undefined ) { 
            returnData.push(jsonData) 
        }
      }

      return returnData

    }, i, sido, sigungu)

    // console.log(infoArr)

    finalData = finalData.concat(infoArr) // for문을 돌면서 한페이지씩 긁어온 내용들을 finalData라는 빈 리스트에 합쳐라
    // concat : push와 비슷하게 array에 있는 내장함수

    // console.log(finalData.length) // 한페이지에 몇개의 내용이 있는지 나옴

    // 다음페이지로 이동
    if ( pageLength != i ) {

      await page.evaluate(function(i, pageSelector) {
        document.querySelector(pageSelector).children[i].click()
      },i, pageSelector)

      // 페이지 로딩 대기
      await page.waitForSelector('#printZone')

    }

  }

  browser.close()

} // End getData

// 9) writerFile 함수 생성 - 디렉토리 생성하기
const writerFile = async function () {

  // 불러온 모든 내용(finalData)을 문자열로 변환시켜라
  // JSON.stringify() : JavaScipt 값이나 객체를 JSON 문자열로 변환
  const stringData = JSON.stringify(finalData)

  // fs.existsSync(path)
  // path <문자열> <URL> <버퍼>
  // 반환값 <부울>
  // true 경로가 있으면 반환 , false 그렇지 않으면 반환
  const exist = fs.existsSync(`./json/${sido}`)

  if (!exist) {

    // fs.mkdir( path[, options], callback )
    // 비동기식으로 디렉토리 생성
    // path <문자열> <버퍼> <URL>
    // options <오브젝트> <정수>
    // - recursive <부울> 기본값 : false
    // - mode <문자열> <정수>
    // callback <기능>
    // - err <오류>
    // - path <문자열>

    //           path               options           callback
    fs.mkdir(`./json/${sido}`, {recursive : true}, function(err) {
      console.log(err)
    })
  }

  const filepath = `./json/${sido}/${sigungu}.json`

  // fs.writeFileSync(file, data[, options])
  // 반환(undefined) 한다 ?
  // file <문자열> <버퍼> <URL> <정수>
  // data <문자열> <버퍼> <유형배열> <데이터뷰> <오브젝트>
  // options <오브젝트> <문자열>
  // - encoding <문자열> <null>
  // - mode <정수>
  // - flag <string>

  await fs.writeFileSync(filepath, stringData)

}

export {
  launch,
  goto,
  checkPopup,
  evalCode,
  evalCity,
  alertClose,
  getpageLength,
  getData,
  writerFile
}
import puppeteer from 'puppeteer-core'
import os from 'os'
import fs, { fdatasync } from 'fs'

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


let browser = null
let page = null 
let infoArr = []


// 브라우저 열기
const launch = async function () {
    browser = await puppeteer.launch(launchConfig)

    const pages = await browser.pages()
    page = pages[0]
}

// URL 연결
const goto = async function (url) {
    return await page.goto(url)
}

// 배너 닫기
const bannerClose = async function () {
    await page.evaluate(function(){
        document.querySelector("#mainPop > div > div.wrap_lower > div.fl_right > button > span").click()
    },)
}

// 멜론차트 클릭
const chartClick = async function () {
    await page.evaluate(function () {
        document.querySelector("#gnb_menu > ul:nth-child(1) > li.nth1 > a").click()
    })
}

// TOP100 데이터 가져오기
const getData = async function () {

    // 페이지 로드
    await page.waitForSelector(`#frm > div > table`)

    // 순위 * 곡 제목 * 가수명 * 앨범명 * 좋아요
    infoArr = await page.evaluate(function () {
        var trArr = document.querySelectorAll("#frm > div > table > tbody tr")

        // 정보 담을 빈리스트
        var returnData = []

        // 정보 긁기
        for (var i=0; i < trArr.length; i++) {

            var currentTr = trArr[i]

            // 순위
            var ranking = currentTr.querySelectorAll('td')[1]?.innerText.replace('\n', '')

            // 곡 제목
            var title =  currentTr.querySelectorAll('td')[5].querySelector('.rank01').innerText

            // 가수명
            var singer =  currentTr.querySelectorAll('td')[5].querySelector('.rank02').innerText

            // 앨범명
            var album =  currentTr.querySelectorAll('td')[6].innerText

            // 좋아요
            var like = currentTr.querySelectorAll('td')[7]?.innerText.replace('\n총건수\n', '')

            var jsonData = { ranking, title, singer, album, like }

            if ( jsonData.ranking != undefined ) {
                returnData.push(jsonData)
            }

        }

        return returnData

    } )

    // console.log(infoArr)

    browser.close()
}

// TOP100 파일 생성
const writerFile = async function () {

    const stringData = JSON.stringify(infoArr)

    const exist = fs.existsSync(`./melon`)

    if (!exist) {
        fs.mkdir(`./melon`, {recursive : true}, function(err){
            console.log(err)
        })
    }

    const filepath = './melon/TOP100.json'

    await fs.writeFileSync(filepath, stringData)
}

export {
    launch,
    goto,
    bannerClose,
    chartClick,
    getData,
    writerFile
}
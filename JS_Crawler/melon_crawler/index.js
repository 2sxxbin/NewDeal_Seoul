import {launch, goto, bannerClose, chartClick, getData, writerFile} from './modules/crawler.js'

async function main () {
    await launch()
    await goto('https://www.melon.com/')
    await bannerClose()
    await chartClick()
    await getData()
    await writerFile()
    process.exit(1)
}

main()
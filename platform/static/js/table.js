
var backupAataArray = [], //接受接口的数据，避免翻页的时候发送多次请求
    dataTotalNum = 0, //数据总条数
    tableNum = 15, //每一页展示的数据条数
    filtrateTable = [], //table每次需要展示的数据，受每页展示条数的限制;
    currentPageNum = 1, //当前所在页码;
    pageFixedNum = 6, //初始化定的页码数量
    dataSet = new Set(); //用集合来存储选中的数据

var setOn = true;
/*初始化表格的函数 S*/
tableInit(user_id);
/*初始化表格的函数 E*/
/*
 * 渲染的table的id 重要，重要。渲染多个table的关键（这个留着我下次再封装吧，突然感觉还要考虑很多东西）
 * jsonData:数据（这里是使用的固定数据，因为ajax本地跨域的问题，所以没有使用ajax传递数据，项目中，需根据实际环境做出调整）
 * 每页展示的数据条数,默认为5 可填写的格式为，number true(为默认)
 * 可修改的每页展示条数，默认为[5,10,20];，填写格式为 array true(为默认)
 * 是否添加翻页选中效果 此处使用了set集合的形式，避免重合，所以请使用唯一标识来区别 true 默认打开，false 关闭
 * */
function tableInit(user_id) {
    var tableNum = 15;
    $(".reminder").remove();
    $.ajax({
        type: 'GET',
        async: false,
        url: "/record/get/",
        data: {"user_id": user_id},
        success: function(rdata){
            backupAataArray = rdata;
            dataTotalNum = backupAataArray.length;
            if (backupAataArray.length > tableNum) {
                filtrateTable = backupAataArray.slice(0, tableNum);
            } else {
                filtrateTable = backupAataArray.slice(0);
            }
        },
        dataType: "json"
    });
    var pages = [15];
    for (var i in pages) {
        var optionPage = $('<option value="' + pages[i] + '">' + pages[i] + '</option>');
        $(".tablesLength").append(optionPage);
    }
    tableCreate();
    /*创建页码*/
    createPages();
};


/*创建table数据*/
function tableCreate() {
    $("#userImportTable>tbody").html("");
    /*创建table里面的数据结构 td的各项值 S*/
    for (var i in filtrateTable) {
        var trHtml = $("<tr>" +
            "<td>" + (filtrateTable[i].record_id || 'N/A') + "</td>" +
            "<td>" + filtrateTable[i].turn + "</td>" +
            "<td>" + (filtrateTable[i].create_time || 'N/A') + "</td>" +
            "<td>" + (filtrateTable[i].end_time || 'N/A') + "</td>" +
            "<td>" +
            "<button class='checkBtn' onclick='pop(&quot;" + filtrateTable[i].record_id + "&quot;)'>详情</button>" +
            "<button class='checkBtn' onclick='check(&quot;" + filtrateTable[i].record_id + "&quot;)'>删除</button>" +
            "</td>" +
            "</tr>");
        //按照集合内容判断现选中的input 翻页保留选中效果
        if (setOn) {
            var checkedTdId = trHtml.find(".ceckedImportTd>input").val();
            var checkedFlg = "";
            if (dataSet.has(checkedTdId)) {
                checkedFlg = "checked";
                trHtml.find(".ceckedImportTd>input").prop("checked", true);
            }
        } else {
            dataSet.clear();
            chooseDataNm();
        }

        /*添加tr内容到table tbody 中*/
        $("#userImportTable>tbody").append(trHtml);
    }
};

/*创建翻页*/
function createPages() {
    /*批量导入用户翻页效果  需要实际数据支持*/
    var pagesNum = Math.ceil(dataTotalNum / tableNum); //计算总页码数
    $(".tablePageinate").html("");
    var pagesButtons = $('<a class="pageinateBtn previous disabled" aria-controls="userImportTable" pageId="0" tabindex="0" previousPage="0" id="pagePrevious">&lt;</a>' +
        '<span class="pageBtnS">' +
        '</span>' +
        '<a class="pageinateBtn next" aria-controls="userImportTable" pageId="2" tabindex="0" nextPage="' + pagesNum + '" id="pageNext">&gt;</a>')
    $(".tablePageinate").append(pagesButtons);
    if (pagesNum <= 0) {
        return false;
    }
    var detailPage = $('<a class="pageinateBtn current" aria-controls="userImportTable" pageId="1" tabindex="0">1</a>');
    $(".tablePageinate").find(".pageBtnS").append(detailPage);

    //初始化页码，包括当页码是在标准页码pageFixedNum之内的页码数，包括产生省略号的页码效果
    if (pagesNum >= 2 && pagesNum <= pageFixedNum) {
        for (var num = 2; num <= pagesNum; num++) {
            var detailPages = $('<a class="pageinateBtn " aria-controls="userImportTable" pageId="' + num + '" tabindex="0">' + num + '</ a>');
            $(".tablePageinate").find(".pageBtnS").append(detailPages);
        }
    }
    if (pagesNum > pageFixedNum) {
        var previousEllipsis = $('<span class="previousEllipsis" hidden="hidden">...</span>');
        $(".tablePageinate").find(".pageBtnS").append(previousEllipsis);
        for (var num = 2; num <= pageFixedNum - 1; num++) {
            var detailPages = $('<a class="pageinateBtn " aria-controls="userImportTable" pageId="' + num + '" tabindex="0">' + num + '</ a>');
            $(".tablePageinate").find(".pageBtnS").append(detailPages);
        }
        for (var num = pageFixedNum; num < pagesNum; num++) {
            var detailPages = $('<a class="pageinateBtn" aria-controls="userImportTable" pageId="' + num + '" tabindex="0" style="display:none">' + num + '</ a>');
            $(".tablePageinate").find(".pageBtnS").append(detailPages);
        }
        var nextEllipsis = $('<span class="nextEllipsis">...</span>');
        $(".tablePageinate").find(".pageBtnS").append(nextEllipsis);
        var detailPages = $('<a class="pageinateBtn" aria-controls="userImportTable" pageId="' + pagesNum + '" tabindex="0" >' + pagesNum + '</ a>');
        $(".tablePageinate").find(".pageBtnS").append(detailPages);
    };


    /*翻页效果*/
    /*前翻页*/
    $("#pagePrevious").bind("click", function () {
        var pageNum = parseInt($(this).attr("pageId"));
        /*前翻页禁用效果*/
        if (pageNum <= 0) {
            $(this).addClass("disabled");
            return false;
        }
        if (pageNum > 0) {
            currentPageNum = pageNum;
            pageNum--;
            $("#pageNext").attr("pageId", pageNum + 2).removeClass("disabled");
            $(this).attr("pageId", pageNum);
            $(".pageBtnS>a").eq(pageNum).addClass("current").siblings().removeClass("current");
            /*获取分页数据*/
            filtrateTable = backupAataArray.slice((currentPageNum - 1) * tableNum, currentPageNum * tableNum);
            tableCreate();

        }

        /*前翻页适当保留6位页码数，前省略点的消失和后省略点的出现的情况*/
        if ($(".pageBtnS>a").eq(pageNum - 1).css("display") == "none" && $(".pageBtnS>a").eq(pageNum - 2).css("display") == "none") {
            $(".pageBtnS>a").eq(pageNum - 1).css("display", "inline-block")
        }
        /*console.log(pageNum);*/
        if (pageNum < 4) {
            $(".previousEllipsis").hide();
            $(".nextEllipsis").show();

            for (var i = 1; i < 4; i++) {
                $(".pageBtnS>a").eq(i).css("display", "inline-block");
            }
            for (var i = 2; i < pagesNum - 4; i++) {
                $(".pageBtnS>a").eq(pagesNum - i).css("display", "none");
            }
            return false;
        }
        if (pagesNum - pageNum > 3) {
            $(".nextEllipsis").show();
            for (var i = pageNum + 1; i < pagesNum - 2; i++) {
                $(".pageBtnS>a").eq(i + 1).css("display", "none");
            }
        }
    })
    /*后翻页*/
    $("#pageNext").bind("click", function () {
        var pageNum = parseInt($(this).attr("pageId"));
        if (pageNum > pagesNum) {
            $(this).addClass("disabled");
            return;
        }
        if (pageNum < pagesNum + 1) {
            $("#pagePrevious").attr("pageId", pageNum - 1).removeClass("disabled");
            currentPageNum = pageNum;
            pageNum++;
            $(this).attr("pageId", pageNum);
            $(".pageBtnS>a").remove("current");
            $(".pageBtnS>a").eq(pageNum - 2).addClass("current").siblings().removeClass("current");

        }

        /*翻页效果产生省略号的延续效果*/
        /*翻页数据修改*/
        /*判断当前页数数据是否为5条满数据，分别的截取方法*/

        if (currentPageNum * tableNum > dataTotalNum) {
            filtrateTable = backupAataArray.slice((currentPageNum - 1) * tableNum);
        } else {
            filtrateTable = backupAataArray.slice((currentPageNum - 1) * tableNum, currentPageNum * tableNum);
        }
        tableCreate();



        /*后翻页的时候 适当保留后五位页码显示，此时后省略号消失。控制前省略号的显示情况，当页码显示数量超过6时，即可产生前省略号*/
        if ($(".pageBtnS>a").eq(pageNum - 1).css("display") == "none" && $(".pageBtnS>a").eq(pageNum).css("display") == "none") {
            $(".pageBtnS>a").eq(pageNum - 1).css("display", "inline-block");
        }
        if (pagesNum - pageNum < 2) {
            $(".nextEllipsis").hide();
            $(".previousEllipsis").show();
            for (var i = 1; i < 5; i++) {
                $(".pageBtnS>a").eq(pagesNum - i).css("display", "inline-block");
            }
            for (var i = 1; i < pagesNum - 5; i++) {
                $(".pageBtnS>a").eq(i).css("display", "none");
            }
            return false;
        }
        if (pageNum >= 6) {
            $(".previousEllipsis").show();
            for (var i = 1; i < pageNum - 3; i++) {
                $(".pageBtnS>a").eq(i).css("display", "none");
            }
        }
    })
    /*自选页码点击翻页事件*/
    $(".pageBtnS .pageinateBtn").bind("click", function () {
        var currentPageNum = parseInt($(this).attr("pageId"));
        /*控制前省略号显示事件，页码在5和倒数第3时触发事件*/
        if (currentPageNum <= 5) {
            $(".previousEllipsis").hide();
            $(".nextEllipsis").show();
            for (var i = 1; i < 5; i++) {
                $(".pageBtnS>a").eq(i).css("display", "inline-block");
            }
            for (var j = 5; j < pagesNum - 1; j++) {
                $(".pageBtnS>a").eq(j).css("display", "none");
            }
        }
        /*前后翻页省略*/
        if (currentPageNum > 4 && pagesNum - currentPageNum >= 3) {
            $(".previousEllipsis").show();
            $(".nextEllipsis").show();
            $(".pageBtnS>a").eq(currentPageNum - 2).css("display", "inline-block");
            $(".pageBtnS>a").eq(currentPageNum).css("display", "inline-block");
            for (var i = 1; i < pageNum - 2; i++) {
                $(".pageBtnS>a").eq(i).css("display", "none");
            }
            for (var j = currentPageNum + 1; j < pagesNum - 1; j++) {
                $(".pageBtnS>a").eq(j).css("display", "none");
            }
        }
        /*后翻省略*/
        if (pagesNum - currentPageNum < 4) {
            $(".nextEllipsis").hide();
            $(".previousEllipsis").show();
            for (var i = 1; i < 6; i++) {
                $(".pageBtnS>a").eq(pagesNum - i).css("display", "inline-block");
            }
            for (var j = 1; j < pagesNum - 5; j++) {
                $(".pageBtnS>a").eq(j).css("display", "none");
            }
        }


        $(".pageBtnS>a").eq(currentPageNum - 1).addClass("current").siblings().removeClass("current");
        if (currentPageNum * tableNum > dataTotalNum) {
            filtrateTable = backupAataArray.slice((currentPageNum - 1) * tableNum);
        } else {
            filtrateTable = backupAataArray.slice((currentPageNum - 1) * tableNum, currentPageNum * tableNum);
        }
        $("#pagePrevious").attr("pageId", currentPageNum - 1);
        $("#pageNext").attr("pageId", currentPageNum + 1);
        if (currentPageNum - 1 > 0) {
            $("#pagePrevious").removeClass("disabled");
        }
        if (currentPageNum + 1 < pagesNum) {
            $("#pageNext").removeClass("disabled");
        }
        tableCreate();
    })
}


/*设置每页显示的条数S*/
function changePages(pageNum) {
    tableNum = pageNum;
    if (backupAataArray.length > tableNum) {
        filtrateTable = backupAataArray.slice(0, tableNum);
    } else {
        filtrateTable = backupAataArray.slice(0);
    }
    tableCreate();
    createPages();
}

/*设置每页显示的条数E*/
/*编辑按钮*/
function check(id) {
    alert("点击id为" + id + "的编辑按钮！");
}

/* 弹窗的 */
function pop(record_id) {
    var pop_data = []
    $.ajax({
        type: 'GET',
        async: false,
        url: "/record/detail/",
        data: {"record_id": record_id},
        success: function(rdata){
            pop_data = rdata;
        },
        dataType: "json"
    });
    pop_text = '<div id="mry-mask" deletes="mry-opo"></div>' +
        '<div id="mry-opo"><div id="mry-opo-title">详情</div><div id="mry-opo-content">' +
        '<a href="javascript:void(0)" deletes="mry-opo"' +
        ' style="position:absolute;right:10px;top:6px;color:#fff;font-size:12px;">X</a>' +
        '<table class="pop-table">';
    for (var i in pop_data){
        pop_text = pop_text + '<tr> <td> ' + pop_data[i] + ' </td></tr>';
    }
    pop_text = pop_text + '</table></div></div>';
    $('body').append(pop_text);
    $('[deletes=mry-opo]').click(function () {
        $('#mry-opo,#mry-mask').remove();
    });
}
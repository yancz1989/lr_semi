#ifndef __STRESSTEST_H__
#define __STRESSTEST_H__

#include "iReadImageApi.h"
#include "iReadAPI.h"
#include "iReadExtAPI.h"
// #include "write_integration_log.h"

#include <string>
#include <cstring>

using namespace std;

typedef unsigned short      WORD;

#define THREAD_COUNT  20
#define RUN_TIME    60*60*24*3
#define MAX_REGION_COUNT  10
#define _MAX_PATH 260

const int MAX_FILE_COUNT = 5000;
string strDataFileName[MAX_FILE_COUNT];
int nFilesize =0;

int nTestTotalCount = 0;

IREAD_UINT nCallbackCount = 1;
IREAD_BOOL bEndFlag = FALSE; 

const char * pcszLibPath = "../Data/ResData";
const char * pcszDataPath = "../Data/TestData";

typedef struct _ThreadInfo 
{
  IREAD_HANDLE engine_handle;
  int nThreadIndex;
  // CWriteIntegrationLog *pWriteLog;
} ThreadInfo;


IREAD_INT paramImageType[] = {
  IREAD_IMAGE_TYPE_NORMAL,
  IREAD_IMAGE_TYPE_SCREEN
};

IREAD_INT paramImageBinarizeMethod[] = {
  IREAD_BINARIZE_GLOBAL,
  IREAD_BINARIZE_ADAPTIVE
};

IREAD_INT paramLayoutMethod[] = {
  IREAD_LAYOUT_NEWSPAPER
};

IREAD_WORD paramRecogLang[] = {
  IREAD_LANGUAGE_ENGLISH,
  IREAD_LANGUAGE_CHINESE_CN,
  IREAD_LANGUAGE_CHINESE_TW,
  IREAD_LANGUAGE_CHINESE_HK
//  IREAD_LANGUAGE_CUSTOM_E13B
};

IREAD_DWORD paramRecogrRange[] = {
  IREAD_RECOG_RANGE_ALL,
  IREAD_RECOG_RANGE_CUSTOM,
  IREAD_RECOG_RANGE_NUMBER,
  IREAD_RECOG_RANGE_UPPERCASE,
  IREAD_RECOG_RANGE_LOWERCASE,
  IREAD_RECOG_RANGE_LETTER,
  IREAD_RECOG_RANGE_ALNUM,
  IREAD_RECOG_RANGE_GBK
};

IREAD_WORD * paramRecogCustomChars[] = {
  (IREAD_WORD *)L"abcde",
  (IREAD_WORD *)L"0123456789",
  (IREAD_WORD *)L"\0"
};

IREAD_INT paramOutputFullHalf[] = {
  IREAD_FH_HALF,
  IREAD_FH_FULL
};

IREAD_INT paramOutputVertPunc[] = {
  IREAD_VP_NO_CHANGE,
  IREAD_VP_TO_HORZ
};

IREAD_INT paramOutputDispCode[] = {
  IREAD_DP_NO_CHANGE,
  IREAD_DP_TO_SIMPLIFIED
};

IREAD_INT paramOutputResultOption[] = {
  IREAD_RESULT_TEXTBUF,
  IREAD_RESULT_NODE,
  IREAD_RESULT_ALL
};

//IREAD_INT paramTextType[] = {
//  IREAD_TEXT_TYPE_SINGLE_CHAR,
//  IREAD_TEXT_TYPE_SINGLE_LINE_HORZ,
//  IREAD_TEXT_TYPE_SINGLE_LINE_VERT,
//  IREAD_TEXT_TYPE_MULTI_LINE_HORZ,
//  IREAD_TEXT_TYPE_MULTI_LINE_VERT
//};

IREAD_INT paramRgnType[] = {
  IREAD_RGNTYPE_AUTOTEXT,
  IREAD_RGNTYPE_HORZTEXT,
  IREAD_RGNTYPE_VERTTEXT,
  IREAD_RGNTYPE_TABLE,
  IREAD_RGNTYPE_GRAPH,    
  IREAD_RGNTYPE_SINGLE_LINE_HORZTEXT,
  IREAD_RGNTYPE_SINGLE_LINE_VERTTEXT,
  IREAD_RGNTYPE_SINGLE_CHAR,
};

IREAD_WORD paramRgnLang[] = {
  IREAD_LANGUAGE_DEFAULT,
  IREAD_LANGUAGE_ENGLISH,
  IREAD_LANGUAGE_CHINESE_CN,
  IREAD_LANGUAGE_CHINESE_TW,
  IREAD_LANGUAGE_CHINESE_HK
//  IREAD_LANGUAGE_CUSTOM_E13B
};

IREAD_BOOL __stdcall iRead_Output_Callback_Function(const IREAD_WORD * lpLineText, IREAD_INT nLen, 
                          IREAD_HANDLE pUserData);

IREAD_BOOL __stdcall iRead_Progress_Callback_Function(IREAD_INT nPercent, IREAD_HANDLE pUserData);

void GetRandSessionParam(IREAD_HANDLE hOCR);
void GetRandomRect(IREAD_RECT *pRect, const IREAD_IMAGE *pImage);
void GetRandomRegion(IREAD_REGION *pRegion, const IREAD_IMAGE *pImage);

typedef struct OCR_PARAM_T
{
  /// 图像类型，对应的值类型为 #IREAD_INT, 取值范围：@ref 图像类型 "IREAD_IMAGE_TYPE_xxx"
  IREAD_INT iREAD_PARAM_IMAGE_TYPE;
  /// 二值化方法，对应值类型为 #IREAD_INT, 取值范围：@ref 二值化方法 "IREAD_BINARIZE_xxx"
  IREAD_INT iREAD_PARAM_IMAGE_BINARIZE_METHOD;
  /// 版面分析方法，对应值类型为 #IREAD_INT, 取值范围：@ref 版面分析方法 "IREAD_LAYOUT_xxx"
  IREAD_INT iREAD_PARAM_LAYOUT_METHOD;
  /// 识别语言，对应值类型为 #IREAD_WORD, 取值范围：@ref 识别语言 "IREAD_LANGUAGE_xxx"(#IREAD_LANGUAGE_DEFAULT 除外)
  IREAD_WORD  iREAD_PARAM_RECOG_LANG;
  /// 识别范围，对应值类型为 #IREAD_DWORD, 取值范围: @ref 识别范围
  /// "IREAD_RECOG_RANGE_xxx". 可以用“|”操作符合并多个识别范围
  IREAD_DWORD iREAD_PARAM_RECOG_RANGE;
  /// 自定义字符集，对应值类型为 #IREAD_WORD *，以L'\0'结尾的UNICODE字符串指针。
  /// 如身份证号码由数字[0-9]和字符'X'构成，可以采用如下两种方式之一：
  ///  1. 先将iREAD_PARAM_RECOG_LANG类型的参数值设为 #IREAD_RECOG_RANGE_CUSTOM, 定义字符串变量 IREAD_WORD wszID[] = L"0123456789X",
  ///  2. 先将iREAD_PARAM_RECOG_LANG类型的参数值设为 #IREAD_RECOG_RANGE_NUMBER | #IREAD_RECOG_RANGE_CUSTOM, 定义字符串变量 IREAD_WORD wszID[] = L"X", 
  /// 
  /// 最后将自定义字符集参数设为字符串首地址wszID
  /// @warning识别范围包括 #IREAD_RECOG_RANGE_CUSTOM 时生效，且引擎直接使用设置时
  /// 的字符串指针，因而需保证在识别期间此指针不被释放
  IREAD_WORD *  iREAD_PARAM_RECOG_CUSTOM_CHARS;
  /// 英文、数字是否全角输出，对应值类型为 #IREAD_INT, 取值范围：@ref 全半角输出控制 "IREAD_FH_xxx"
  IREAD_INT   iREAD_PARAM_OUTPUT_FULL_HALF;
  /// 竖排符号是否转换为横排，对应值类型为 #IREAD_INT, 取值范围：@ref 竖排符号输出控制 "IREAD_VP_xxx"
  IREAD_INT   iREAD_PARAM_OUTPUT_VERT_PUNC;
  /// 输出结果是否进行简繁转换，对应值类型为 #IREAD_INT, 取值范围：@ref 简繁体输出控制 "IREAD_DP_xxx"
  IREAD_INT   iREAD_PARAM_OUTPUT_DISP_CODE;
  /// 获取结果控制，对应值类型为 #IREAD_INT, 取值范围：@ref 最终获取结果控制 "IREAD_RESULT_xxx"
  IREAD_INT   iREAD_PARAM_OUTPUT_RESULT_OPTION;
}
OCR_PARAM;

typedef struct USER_PARAM_T
{
  // CWriteIntegrationLog *pWriteLog;
  char szImgFile[_MAX_PATH];
  IREAD_IMAGE * pImage;
  OCR_PARAM ocrParam;
  IREAD_REGION * pRegions;
  int nRegionCount;
}
USER_PARAM;

#endif
